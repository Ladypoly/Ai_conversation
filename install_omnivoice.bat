@echo off
echo ============================================
echo   Installing OmniVoice TTS Engine
echo ============================================
echo.

REM Activate virtual environment
if exist "venv_voice_ai\Scripts\activate.bat" (
    call venv_voice_ai\Scripts\activate.bat
    echo Activated venv_voice_ai
) else (
    echo WARNING: Virtual environment not found, using system Python
)

echo.
echo Installing OmniVoice...
pip install omnivoice

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo OmniVoice should now be available as a TTS
echo engine in the Voice Chat AI launcher.
echo.
echo Select "OmniVoice" from the TTS Engine
echo dropdown to use it.
echo.
echo Features:
echo   - 600+ languages
echo   - Voice cloning from reference audio
echo   - Voice design via text attributes
echo     (e.g. "young female, warm, American accent")
echo.
pause
