# Real-Time Local Conversational AI Setup Plan

## Progress Tracker
> **Last Updated:** 2026-03-23
> **Current Step:** COMPLETE
> **Status:** Ready to run

| Step | Status | Notes |
|------|--------|-------|
| 1. Install Python 3.12 | DONE | Python 3.12.0 available |
| 2. Create Virtual Environment | DONE | venv_voice_ai created |
| 3. Install MOSS-TTS | DONE | torch 2.9.1+cu128, transformers 5.0 |
| 4. Install WhisperX | DONE | whisperx 3.8.2, faster-whisper |
| 5. Install Dependencies | DONE | sounddevice, keyboard |
| 6. Verify Ollama + Qwen3.5 | DONE | qwen3.5:4b available |
| 7. Create Integration Scripts | DONE | config.py, audio_utils.py, voice_chat.py |
| 8. Test Pipeline | DONE | All imports verified working |

**Note:** Pip shows version conflict warnings but all components work at runtime.

---

## Overview
Set up a local voice-based conversational AI with:
- **MossTTSRealtime** (1.7B) - Text-to-Speech (180ms latency)
- **WhisperX** - Speech-to-Text (faster-whisper backend)
- **Qwen3.5:4b** - LLM via Ollama

All running on RTX 4090 (24GB VRAM) without offloading.

## System Info
- GPU: RTX 4090 (24GB VRAM, Compute 8.9)
- Current Python: 3.11.4
- Platform: Windows
- Working Directory: `C:\Tools\AI_Code_Projects\Qwen3-TTS`

## Estimated VRAM Usage
| Component | VRAM |
|-----------|------|
| MossTTSRealtime (1.7B) | ~3-4 GB |
| WhisperX (large-v3) | ~3-5 GB |
| Qwen3.5:4b | ~3.4 GB |
| **Total** | **~10-13 GB** |

---

## Step 1: Install Python 3.12
MOSS-TTS requires Python 3.12. Install if not available.

```powershell
# Check if Python 3.12 is available
py -3.12 --version

# If not installed, download from python.org or use winget:
winget install Python.Python.3.12
```

---

## Step 2: Create Virtual Environment

```powershell
cd C:\Tools\AI_Code_Projects\Qwen3-TTS
py -3.12 -m venv venv_voice_ai
.\venv_voice_ai\Scripts\activate
python -m pip install --upgrade pip
```

---

## Step 3: Install MOSS-TTS

```powershell
# Clone MOSS-TTS repository
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS

# Install with PyTorch CUDA 12.8 support + FlashAttention
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,flash-attn]"

# If RAM limited during compilation:
# set MAX_JOBS=4
# pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,flash-attn]"

cd ..
```

---

## Step 4: Install WhisperX

```powershell
pip install whisperx
```

Note: WhisperX uses faster-whisper backend (4x faster, less memory).

---

## Step 5: Install Additional Dependencies

```powershell
pip install sounddevice numpy scipy pyaudio requests keyboard silero-vad
```

- `sounddevice` / `pyaudio` - Audio I/O
- `keyboard` - Push-to-talk key detection
- `silero-vad` - Voice Activity Detection (lightweight, runs on CPU)

---

## Step 6: Setup Ollama with Qwen3.5

```powershell
# Verify Ollama is installed and running
ollama --version

# Model already downloaded: qwen3.5:4b
# Just verify it's available:
ollama list
```

---

## Step 7: Create Integration Script

Create `voice_chat.py` with the following architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Microphone    │────▶│    WhisperX     │────▶│   Qwen3.5:4b    │
│   (Audio In)    │     │   (STT)         │     │   (Ollama LLM)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│    Speaker      │◀────│ MossTTSRealtime │◀─────────────┘
│   (Audio Out)   │     │   (TTS)         │
└─────────────────┘     └─────────────────┘
```

### Script Components:
1. **Audio capture** - Record from microphone using sounddevice
2. **STT** - Transcribe audio with WhisperX
3. **LLM** - Send text to Ollama API (localhost:11434)
4. **TTS** - Generate speech with MossTTSRealtime
5. **Playback** - Play audio response

### Input Modes (Toggle with key):
- **Push-to-talk (PTT)** - Hold SPACE to record, release to process
- **Voice Activity Detection (VAD)** - Auto-detect speech start/stop using silero-vad

### Key Files to Create:
- `voice_chat.py` - Main conversation loop with mode toggle
- `config.py` - Settings (model paths, audio params, VAD sensitivity)
- `audio_utils.py` - Audio capture, VAD, and playback utilities

---

## Step 8: Test the Pipeline

1. Test each component individually:
   - WhisperX transcription
   - Ollama API response
   - MOSS-TTS audio generation

2. Run full conversation loop

---

## Potential Issues & Solutions

| Issue | Solution |
|-------|----------|
| Python 3.12 not found | Install via winget or python.org |
| CUDA version mismatch | Install CUDA 12.8 toolkit |
| FlashAttention compile fails | Use `MAX_JOBS=4`, ensure VS Build Tools |
| Ollama not running | Start with `ollama serve` |
| Audio device issues | Check sounddevice device list |

---

## Files Created
1. `setup_install.bat` - **Run this first** to install all dependencies
2. `run_voice_chat.bat` - Launch the Voice Chat UI
3. `run_direct.bat` - Launch directly (skip UI)
4. `launcher.py` - GUI with settings, status, and conversation log
5. `voice_chat.py` - Main conversation script
6. `config.py` - Configuration settings
7. `audio_utils.py` - Audio recording, VAD, playback utilities

## UI Features

### Chat Tab
- **Voice Level Indicator** - Real-time mic level bar (green/orange/red)
- **Fast Mode** - Skip LLM thinking for faster responses
- **Voice Selection** - Dropdown to select TTS voice (default + custom + trained)
- **LLM Selection** - Dropdown to select Ollama model with refresh button
- **Mode Toggle** - Switch between Push-to-Talk and VAD modes

### Train Voice Tab
- **Folder Selection** - Browse for training data folder
- **Sample Count** - Choose how many samples to use (quality vs speed)
- **Epochs** - Training iterations (1-10)
- **Progress Bar** - Real-time training progress
- **Log Output** - Live training logs

## Training a Custom Voice

1. Prepare your training data folder:
   ```
   my_voice_training/
   ├── sample1.wav
   ├── sample1.txt  (transcript: "Hello, this is a test.")
   ├── sample2.mp3
   ├── sample2.txt
   └── ...
   ```

2. Go to the **Train Voice** tab in the launcher
3. Click **Browse** and select your training folder
4. Click **Scan Folder** to detect audio+transcript pairs
5. Enter a **Voice Name** for your trained voice
6. Adjust **Samples to use** and **Epochs** as needed
7. Click **Start Training**
8. Once complete, your voice appears in the Chat tab's voice dropdown as "Trained: <name>"

**Notes:**
- More samples = better quality but longer training
- More epochs = better fit but risk of overfitting
- Training saves to `voices/<name>/` folder

---

## How to Run

### Option 1: With Launcher UI
Double-click `run_voice_chat.bat` or run:
```
.\run_voice_chat.bat
```

### Option 2: Direct (skip UI)
Double-click `run_direct.bat` or run:
```
.\run_direct.bat
```

### Option 3: Command Line
```
.\venv_voice_ai\Scripts\activate
python voice_chat.py
```

---

## Controls
- **SPACE** - Hold to speak (Push-to-Talk mode)
- **TAB** - Toggle between PTT and VAD mode
- **ESC** - Exit

---

## Continuation Notes
> If context runs out, a new Claude instance should:
> 1. Read this file first to understand progress
> 2. Check the Progress Tracker table above
> 3. Continue from the current step
> 4. Update the Progress Tracker after completing each step
