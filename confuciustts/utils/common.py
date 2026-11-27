
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import safetensors.torch


def get_default_device() -> str:
    """Pick the best available accelerator: CUDA > Intel XPU (Arc) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def resolve_device(device: Optional[str]) -> str:
    """Map None / "auto" to the best available device, pass anything else through."""
    if device is None or device == "auto":
        return get_default_device()
    return device


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

