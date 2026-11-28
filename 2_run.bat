@echo off
setlocal EnableExtensions
REM ==========================================================================
REM  Confucius4-TTS - JALANKAN web UI (setelah 1_setup.bat selesai).
REM  Argumen tambahan diteruskan ke webui.py, misalnya:
REM      2_run.bat --host 0.0.0.0     buka untuk perangkat lain di jaringan
REM      2_run.bat --port 7861
REM ==========================================================================
cd /d "%~dp0"
title Confucius4-TTS - Web UI
set "PY=.venv\Scripts\python.exe"
set "PORT=7860"

if not exist "%PY%" goto :need_setup
if not exist ".venv\.confucius_setup_ok" goto :need_setup

REM Hanya satu proses model yang muat di VRAM 12 GB.
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 goto :already

set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_OFFLINE=1"

echo.
echo ============================================================
echo   Confucius4-TTS Web UI
echo   Memuat model ke GPU, sekitar 30-60 detik.
echo   Browser akan terbuka otomatis di http://127.0.0.1:%PORT%
echo.
echo   Biarkan jendela ini terbuka selama dipakai.
echo   Untuk berhenti: tutup jendela ini atau tekan Ctrl+C.
echo ============================================================
echo.

"%PY%" -W ignore webui.py --host 127.0.0.1 --port %PORT% --inbrowser %*

echo.
echo Web UI berhenti.
pause
exit /b 0

:need_setup
echo.
echo Setup belum lengkap. Jalankan 1_setup.bat dulu, lalu buka 2_run.bat lagi.
echo.
pause
exit /b 1

:already
echo Web UI sudah berjalan di jendela lain. Membuka browser ...
start "" "http://127.0.0.1:%PORT%"
timeout /t 4 >nul
exit /b 0
