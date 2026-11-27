"""Static-KV-cache + device-graph decoding for the T2S GPT-2 stack.

Why: on Intel Arc (XPU) under Windows every kernel launch costs ~30-60 us of
host time, and one decode step of the 24-layer GPT-2 issues ~600 kernels, so
token generation is launch-bound (~17-20 ms/step). Recording one decode step
into an XPU graph and replaying it removes that overhead (~6.5 ms/step on B580).

How it plugs into HuggingFace ``generate``:
- prefill (first call, full prefix): run the normal transformer, then copy the
  returned KV tensors into a preallocated static cache;
- each later step: the token embedding is written into a static input buffer and
  a captured graph runs all layers + final norms + semantic head;
- beam search calls ``past_key_values.reorder_cache(beam_idx)`` on our cache
  object, which reorders the static cache in place.

The sampling / beam-search algorithm is untouched: only the model forward is
replaced, and outputs match the eager path to fp32 rounding (~1e-6).

Assumes an all-ones attention mask (no left padding), which holds for
``Text2Semantic.generate``.
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def graph_decode_supported(device: torch.device) -> bool:
    if os.environ.get("CONFUCIUS_GRAPH_DECODE", "1") == "0":
        return False
    if device.type != "xpu":  # only tested on Intel XPU; CUDA users have vLLM
        return False
    return hasattr(torch, "xpu") and hasattr(torch.xpu, "XPUGraph")


class StaticKVCache:
    """Handle passed through HF ``generate`` as ``past_key_values``."""

    def __init__(self, decoder: "GraphDecoder"):
        self.decoder = decoder

    def __bool__(self):
        return True

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.decoder.length

    def reorder_cache(self, beam_idx: torch.Tensor):
        self.decoder.reorder(beam_idx)


class GraphDecoder:
    """Owns the static KV cache and one captured graph per (batch, context bucket)."""

    BUCKET = 256  # attention context is padded to a multiple of this

    def __init__(self, model, max_len: int):
        self.model = model
        tr = model.transformer
        p = next(tr.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.n_layer = len(tr.h)
        self.n_embd = tr.embed_dim
        self.n_head = tr.h[0].attn.num_heads
        self.head_dim = self.n_embd // self.n_head
        self.max_len = ((max_len + self.BUCKET - 1) // self.BUCKET) * self.BUCKET
        self.batch = 0
        self.length = 0
        self.graphs: Dict[int, Tuple[object, torch.Tensor]] = {}
        acc = getattr(torch, self.device.type)
        self._acc = acc
        self._pool = acc.graph_pool_handle()
        self._arange = torch.arange(self.max_len, device=self.device)
        self._pos = torch.zeros(1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ buffers
    def _alloc(self, batch: int):
        if batch == self.batch:
            return
        # Captured graphs reference the old buffers: drop them, and start a fresh
        # memory pool (re-capturing into a pool whose graphs were freed asserts).
        self.graphs = {}
        self.k = self.v = self.x_in = None
        self._pool = self._acc.graph_pool_handle()
        shape = (self.n_layer, batch, self.n_head, self.max_len, self.head_dim)
        self.k = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.x_in = torch.zeros(batch, 1, self.n_embd, device=self.device, dtype=self.dtype)
        self.batch = batch

    # -------------------------------------------------------------- one step
    def _step(self, ctx: int) -> torch.Tensor:
        """Decode one token for every row; attention looks at cache[:ctx]."""
        B, H, D = self.batch, self.n_head, self.head_dim
        tr = self.model.transformer
        pos = self._pos
        mask = (self._arange[:ctx] <= pos).view(1, 1, 1, ctx)
        x = self.x_in
        for i, blk in enumerate(tr.h):
            h = blk.ln_1(x)
            q, k, v = blk.attn.c_attn(h).split(self.n_embd, dim=2)
            q = q.view(B, 1, H, D).transpose(1, 2)
            k = k.view(B, 1, H, D).transpose(1, 2)
            v = v.view(B, 1, H, D).transpose(1, 2)
            self.k[i].index_copy_(2, pos, k)
            self.v[i].index_copy_(2, pos, v)
            a = F.scaled_dot_product_attention(
                q, self.k[i, :, :, :ctx], self.v[i, :, :, :ctx], attn_mask=mask
            )
            x = x + blk.attn.c_proj(a.transpose(1, 2).reshape(B, 1, self.n_embd))
            x = x + blk.mlp(blk.ln_2(x))
        x = self.model.final_norm(tr.ln_f(x))
        return self.model.semantic_head(x)

    def _graph_for(self, ctx: int):
        if ctx in self.graphs:
            return self.graphs[ctx]
        acc = self._acc
        # Warm up on a side stream (required before capture). The writes it makes
        # land at position `self._pos`, which the real step overwrites anyway.
        side = acc.Stream()
        side.wait_stream(acc.current_stream())
        with acc.stream(side):
            for _ in range(2):
                self._step(ctx)
        acc.current_stream().wait_stream(side)
        g = getattr(acc, "XPUGraph" if self.device.type == "xpu" else "CUDAGraph")()
        with acc.graph(g, pool=self._pool):
            out = self._step(ctx)
        self.graphs[ctx] = (g, out)
        return self.graphs[ctx]

    # ------------------------------------------------------------ public API
    def prefill(self, past_key_values) -> StaticKVCache:
        """Copy the prefix KV cache (legacy tuples or a Cache) into the static cache."""
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        k0 = past_key_values[0][0]
        batch, length = k0.shape[0], k0.shape[2]
        if length >= self.max_len:
            raise ValueError(f"prefix length {length} exceeds static cache {self.max_len}")
        self._alloc(batch)
        for i, (k, v) in enumerate(past_key_values):
            self.k[i, :, :, :length].copy_(k)
            self.v[i, :, :, :length].copy_(v)
        self.length = length
        return StaticKVCache(self)

    def decode(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """inputs_embeds: (B, 1, D) -> logits (B, 1, V)."""
        if self.length >= self.max_len:
            raise ValueError("static KV cache is full")
        self.x_in.copy_(inputs_embeds)
        self._pos.fill_(self.length)
        ctx = min(((self.length + 1 + self.BUCKET - 1) // self.BUCKET) * self.BUCKET, self.max_len)
        g, out = self._graph_for(ctx)
        g.replay()
        self.length += 1
        return out.clone()

    def reorder(self, beam_idx: torch.Tensor):
        # Reorder up to the next bucket boundary instead of exactly `length`: the
        # temporary from index_select then has one of a few fixed sizes, so the caching
        # allocator reuses its block. With an exact (growing every step) size, each step
        # needed a slightly bigger block and the reserved pool grew by GBs over a long
        # utterance; on Arc/Windows that silently spills into shared system memory
        # instead of raising OOM, and generation slows to a crawl or the device is lost.
        # Positions past `length` are never attended, so reordering them is harmless.
        L = min(((self.length + self.BUCKET - 1) // self.BUCKET) * self.BUCKET, self.max_len)
        beam_idx = beam_idx.to(self.device)
        self.k[:, :, :, :L] = self.k[:, :, :, :L].index_select(1, beam_idx)
        self.v[:, :, :, :L] = self.v[:, :, :, :L].index_select(1, beam_idx)
