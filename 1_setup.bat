@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  Confucius4-TTS - SETUP (jalankan sekali, atau lagi kalau ada masalah)
REM  Cek sistem -> pasang uv + Python 3.10 + PyTorch XPU + dependensi
REM  -> unduh model -> tes generate. Aman dijalankan berulang kali.
REM ==========================================================================
cd /d "%~dp0"
title Confucius4-TTS - Setup

set "TORCH_VER=2.14.1"
set "TORCHAUDIO_VER=2.11.0"
set "XPU_INDEX=https://download.pytorch.org/whl/xpu"
set "PY=.venv\Scripts\python.exe"
set "MARKER=.venv\.confucius_setup_ok"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
REM uv cache and .venv are often on different drives; copy instead of hardlink, no warning.
set "UV_LINK_MODE=copy"

echo.
echo ============================================================
echo   Confucius4-TTS - Setup untuk Intel Arc - Windows
echo ============================================================

REM --- 1. System check ----------------------------------------------------------
echo.
echo [1/6] Mengecek sistem ...
powershell -NoProfile -ExecutionPolicy Bypass -File "tools\check_system.ps1" -RepoDir "%~dp0."
if errorlevel 2 goto :sys_warn
if errorlevel 1 goto :fail
goto :sys_ok
:sys_warn
echo.
echo Ada peringatan di atas.
if defined NOPAUSE goto :sys_ok
choice /c YN /m "Tetap lanjutkan setup"
if errorlevel 2 goto :cancel
:sys_ok

REM --- 2. uv ------------------------------------------------------------------
echo.
echo [2/6] Menyiapkan uv - pengelola Python ...
where uv >nul 2>nul
if not errorlevel 1 goto :uv_ok
if exist "%USERPROFILE%\.local\bin\uv.exe" goto :uv_path
echo   uv belum ada, memasang dari astral.sh ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
:uv_path
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 goto :uv_fail
:uv_ok
for /f "delims=" %%V in ('uv --version') do echo   %%V

REM --- 3. Python 3.10 venv ----------------------------------------------------
echo.
echo [3/6] Menyiapkan Python 3.10 di folder .venv ...
if exist "%PY%" (
    "%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo   .venv rusak atau versi Python salah, dibuat ulang ...
        rmdir /s /q .venv
    )
)
if not exist "%PY%" uv venv --python 3.10 .venv
if errorlevel 1 goto :fail
del "%MARKER%" >nul 2>nul

REM --- 4. PyTorch XPU ---------------------------------------------------------
echo.
echo [4/6] Memasang PyTorch %TORCH_VER% untuk Intel XPU - unduhan pertama +-1 GB ...
uv pip install --python "%PY%" torch==%TORCH_VER% --index-url %XPU_INDEX%
if errorlevel 1 goto :fail
REM torchaudio XPU terbaru mengunci torch 2.11; bagian yang dipakai murni PyTorch.
uv pip install --python "%PY%" --no-deps torchaudio==%TORCHAUDIO_VER% --index-url %XPU_INDEX%
if errorlevel 1 goto :fail

REM --- 5. Other dependencies --------------------------------------------------
echo.
echo [5/6] Memasang dependensi lain ...
uv pip install --python "%PY%" -r requirements_xpu.txt
if errorlevel 1 goto :fail
"%PY%" -c "import torch, sys; v = torch.__version__; print('  torch', v); sys.exit(0 if v.endswith('+xpu') else 1)"
if errorlevel 1 goto :torch_fail

REM --- 6. Models + self-test --------------------------------------------------
echo.
echo [6/6] Mengunduh model dan tes generate - pertama kali +-5 GB ...
"%PY%" -W ignore tools\selftest.py
if errorlevel 1 goto :fail

echo %TORCH_VER%> "%MARKER%"
echo.
echo ============================================================
echo   SETUP SELESAI. Semua siap.
echo   Untuk memakai: klik 2_run.bat
echo   Contoh hasil tes: outputs\selftest.wav
echo ============================================================
if not defined NOPAUSE pause
exit /b 0

:uv_fail
echo [GAGAL] uv tidak bisa dipasang. Coba manual: winget install astral-sh.uv
goto :fail
:torch_fail
echo [GAGAL] PyTorch yang terpasang bukan versi XPU. Jalankan 1_setup.bat lagi.
goto :fail
:cancel
echo Setup dibatalkan.
if not defined NOPAUSE pause
exit /b 1
:fail
echo.
echo ============================================================
echo   [GAGAL] Setup berhenti karena error di atas.
echo   Cek koneksi internet, lalu jalankan 1_setup.bat lagi.
echo ============================================================
if not defined NOPAUSE pause
exit /b 1
