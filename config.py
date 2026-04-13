"""
Configuration for Voice Chat AI
"""

# Audio settings
SAMPLE_RATE = 16000  # WhisperX expects 16kHz
CHANNELS = 1
CHUNK_DURATION = 0.5  # seconds per audio chunk for VAD

# Model settings
WHISPER_MODEL = "large-v3"  # or "medium", "small" for faster inference
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434"

TTS_MODEL = "OpenMOSS-Team/MOSS-TTS-Realtime"
TTS_ENGINE = "MOSS-TTS"  # Options: "MOSS-TTS", "Qwen3-TTS", "OmniVoice"
QWEN3_TTS_SPEAKER = "Chelsie"  # For Qwen3-TTS: Chelsie, Ethan, Airi, Zara, Rafaela, Sky, Theo, Nova, Harper
OMNIVOICE_LANGUAGE = "en"
OMNIVOICE_SPEED = 1.0

# Voice Activity Detection settings
VAD_THRESHOLD = 0.5  # Silero VAD threshold (0.0 - 1.0)
VAD_MIN_SPEECH_DURATION = 0.25  # Minimum speech duration in seconds
VAD_MIN_SILENCE_DURATION = 0.5  # Silence duration to end recording

# Push-to-talk settings
PTT_KEY = "space"  # Key to hold for push-to-talk

# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses concise and conversational -
aim for 1-3 sentences unless more detail is needed. Be friendly and natural."""

# Input mode: "ptt" (push-to-talk) or "vad" (voice activity detection)
DEFAULT_INPUT_MODE = "ptt"
MODE_TOGGLE_KEY = "tab"  # Key to toggle between PTT and VAD modes
