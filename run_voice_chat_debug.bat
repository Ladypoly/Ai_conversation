@echo off
title Voice Chat AI (Debug)
cd /d "%~dp0"

REM Debug launcher - shows console output for troubleshooting

if not exist "venv_voice_ai\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_install.bat first.
    pause
    exit /b 1
)

set "PATH=%~dp0venv_voice_ai\Scripts;%PATH%"

if exist "%USERPROFILE%\miniconda3\Library\bin\avcodec-58.dll" (
    set "PATH=%USERPROFILE%\miniconda3\Library\bin;%PATH%"
)
if exist "%USERPROFILE%\Documents\ffmpeg-7.1-full_build\bin\ffmpeg.exe" (
    set "PATH=%USERPROFILE%\Documents\ffmpeg-7.1-full_build\bin;%PATH%"
)

REM Run with console visible for debugging
"venv_voice_ai\Scripts\python.exe" launcher.py
pause
