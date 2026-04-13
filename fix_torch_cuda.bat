@echo off
echo ============================================
echo   Fixing PyTorch CUDA Installation
echo ============================================
echo.
echo OmniVoice installed CPU-only PyTorch.
echo Reinstalling CUDA version...
echo.

REM Activate virtual environment
if exist "venv_voice_ai\Scripts\activate.bat" (
    call venv_voice_ai\Scripts\activate.bat
    echo Activated venv_voice_ai
) else (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

echo.
echo Current PyTorch version:
python -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo.
echo Reinstalling PyTorch with CUDA 12.8 support...
pip install torch==2.9.1+cu128 torchaudio==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128

echo.
echo Verifying installation:
python -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo ============================================
echo   Done! PyTorch CUDA should be restored.
echo ============================================
pause
