@echo off
setlocal enabledelayedexpansion
title Voice Chat AI - Setup & Installation
cd /d "%~dp0"

echo ========================================
echo    Voice Chat AI - Setup Installer
echo ========================================
echo.
echo This script will install all dependencies.
echo Make sure you have:
echo   - Python 3.12 installed
echo   - NVIDIA GPU with CUDA support
echo   - Internet connection
echo.
pause

echo.
echo [1/8] Checking Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 not found!
    echo Please install Python 3.12 from https://python.org
    echo Or run: winget install Python.Python.3.12
    pause
    exit /b 1
)
echo Python 3.12 found!

echo.
echo [2/8] Creating virtual environment...
if exist "venv_voice_ai" (
    echo Virtual environment already exists. Skipping...
) else (
    py -3.12 -m venv venv_voice_ai
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created!
)

echo.
echo [3/8] Upgrading pip...
"venv_voice_ai\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: pip upgrade had issues, continuing anyway...
)

echo.
echo [4/8] Cloning MOSS-TTS repository...
if exist "MOSS-TTS" (
    echo MOSS-TTS already cloned. Skipping...
) else (
    git clone https://github.com/OpenMOSS/MOSS-TTS.git
    if errorlevel 1 (
        echo ERROR: Failed to clone MOSS-TTS!
        echo Make sure Git is installed.
        pause
        exit /b 1
    )
    echo MOSS-TTS cloned!
)

echo.
echo [5/8] Installing MOSS-TTS with PyTorch CUDA 12.8...
echo This may take 10-15 minutes (downloading ~3GB PyTorch)...
cd MOSS-TTS
"..\venv_voice_ai\Scripts\pip.exe" install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
if errorlevel 1 (
    echo WARNING: MOSS-TTS installation had some issues, continuing...
)
cd ..
echo MOSS-TTS installed!

echo.
echo [6/8] Installing WhisperX...
"venv_voice_ai\Scripts\pip.exe" install whisperx
if errorlevel 1 (
    echo WARNING: WhisperX installation had some issues, continuing...
)
echo WhisperX installed!

echo.
echo [7/8] Fixing dependency versions...
echo Reinstalling correct PyTorch and transformers versions...
"venv_voice_ai\Scripts\pip.exe" install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128 torchaudio==2.9.1+cu128 numpy==2.1.0 transformers==5.0.0
if errorlevel 1 (
    echo WARNING: Version fix had issues, continuing...
)

echo.
echo [8/8] Installing additional dependencies...
"venv_voice_ai\Scripts\pip.exe" install sounddevice keyboard
if errorlevel 1 (
    echo WARNING: Some dependencies failed, continuing...
)

echo.
echo ========================================
echo        Installation Complete!
echo ========================================
echo.
echo Testing installation...
"venv_voice_ai\Scripts\python.exe" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
"venv_voice_ai\Scripts\python.exe" -c "import whisperx; print('WhisperX: OK')"
"venv_voice_ai\Scripts\python.exe" -c "import sounddevice; print('SoundDevice: OK')"

echo.
echo ========================================
echo IMPORTANT NOTES:
echo ========================================
echo.
echo 1. First run will download AI models (~7GB total):
echo    - WhisperX large-v3 (~3GB)
echo    - MOSS-TTS-Realtime (~3GB)
echo    - MOSS-Audio-Tokenizer (~500MB)
echo.
echo 2. Make sure Ollama is running with qwen3.5:4b:
echo    ollama serve
echo    ollama pull qwen3.5:4b
echo.
echo 3. FFmpeg warning can be ignored - WhisperX works without it.
echo.
echo To start the Voice Chat AI, run:
echo    run_voice_chat.bat
echo.
pause
