@echo off
title Voice Chat AI
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv_voice_ai\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_install.bat first.
    pause
    exit /b 1
)

REM Add venv Scripts to PATH (for SoX and other tools)
set "PATH=%~dp0venv_voice_ai\Scripts;%PATH%"

REM Add FFmpeg to PATH for voice cloning support (torchcodec needs FFmpeg DLLs)
REM Try miniconda's FFmpeg first (has shared libraries)
if exist "%USERPROFILE%\miniconda3\Library\bin\avcodec-58.dll" (
    set "PATH=%USERPROFILE%\miniconda3\Library\bin;%PATH%"
)
REM Also try standard FFmpeg locations
if exist "%USERPROFILE%\Documents\ffmpeg-7.1-full_build\bin\ffmpeg.exe" (
    set "PATH=%USERPROFILE%\Documents\ffmpeg-7.1-full_build\bin;%PATH%"
)

REM Run launcher (use python.exe for keyboard module compatibility)
"venv_voice_ai\Scripts\python.exe" launcher.py
