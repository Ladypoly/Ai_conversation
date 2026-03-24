@echo off
title Voice Chat AI - Direct Mode
cd /d "%~dp0"

echo ========================================
echo     Voice Chat AI - Direct Mode
echo ========================================
echo.
echo Skipping launcher UI, using default/saved settings...
echo.

REM Check if venv exists
if not exist "venv_voice_ai\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run the setup first.
    pause
    exit /b 1
)

REM Run voice chat directly
"venv_voice_ai\Scripts\python.exe" voice_chat.py

echo.
echo Voice Chat AI has exited.
pause
