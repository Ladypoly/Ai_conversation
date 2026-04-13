"""
Voice Chat AI Launcher
UI that stays open during conversation, shows status and transcripts
"""

# Suppress warnings before importing heavy libraries
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import queue
import sys
import json
import time
import requests
import re
import subprocess

CONFIG_FILE = "user_config.json"


def create_section(parent, title):
    """Create a labeled section frame (replaces ttk.LabelFrame for CTk)"""
    frame = ctk.CTkFrame(parent)
    ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=13, weight="bold")).pack(
        anchor="w", padx=12, pady=(8, 2))
    inner = ctk.CTkFrame(frame, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=10, pady=(0, 8))
    return frame, inner

# TTS Engine options
TTS_ENGINE_OPTIONS = ["MOSS-TTS", "Qwen3-TTS", "OmniVoice"]
QWEN3_SPEAKERS = ["serena", "aiden", "dylan", "eric", "ono_anna", "ryan", "sohee", "uncle_fu", "vivian"]
OMNIVOICE_LANGUAGES = [
    "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar",
    "it", "nl", "pl", "tr", "vi", "th", "id", "hi", "sv", "cs"
]

DEFAULT_PERSONAS = {
    "Helpful Assistant": "You are a helpful voice assistant. Keep your responses concise and conversational - aim for 1-3 sentences unless more detail is needed. Be friendly and natural.",
    "Coding Buddy": "You are a friendly coding assistant. Help with programming questions, explain concepts simply, and suggest solutions. Keep responses brief and conversational since this is a voice interface.",
    "Language Tutor": "You are a patient language tutor. Help practice conversation, correct mistakes gently, and explain grammar when asked. Speak naturally and encourage the learner.",
    "Storyteller": "You are a creative storyteller. Tell engaging short stories, continue narratives, and bring characters to life with expressive speech. Keep each response to a few sentences to maintain flow.",
    "Trivia Host": "You are an enthusiastic trivia game host. Ask interesting questions, give hints when needed, and celebrate correct answers. Keep the energy fun and engaging.",
    "Custom": ""
}

# Voice options - auto-discovered from MOSS-TTS audio folder + VoiceSamples
VOICE_AUDIO_DIR = "MOSS-TTS/moss_tts_realtime/audio"
VOICE_SAMPLES_DIR = "VoiceSamples"
OUTPUT_DIR = "Output"

def get_voice_options():
    """Scan audio folders and build voice options dict"""
    options = {}
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    # Scan MOSS-TTS reference audio
    if os.path.exists(VOICE_AUDIO_DIR):
        for f in sorted(os.listdir(VOICE_AUDIO_DIR)):
            if f.lower().endswith(audio_extensions):
                name = os.path.splitext(f)[0]
                display_name = name.replace('_', ' ').title()
                options[display_name] = os.path.join(VOICE_AUDIO_DIR, f)

    # Scan VoiceSamples folder
    if os.path.exists(VOICE_SAMPLES_DIR):
        for f in sorted(os.listdir(VOICE_SAMPLES_DIR)):
            if f.lower().endswith(audio_extensions):
                name = os.path.splitext(f)[0]
                display_name = f"Sample: {name}"
                options[display_name] = os.path.join(VOICE_SAMPLES_DIR, f)

    # Add special options at the end
    options["Random (No Clone)"] = None
    options["Custom Audio File..."] = "custom"

    return options

VOICE_OPTIONS = get_voice_options()

TRAINED_VOICES_DIR = "voices"


def get_trained_voices():
    """Scan voices/ folder for trained voice models."""
    voices = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    voices_dir = os.path.join(base_dir, TRAINED_VOICES_DIR)

    if not os.path.exists(voices_dir):
        return voices

    for voice_name in os.listdir(voices_dir):
        voice_path = os.path.join(voices_dir, voice_name)
        metadata_path = os.path.join(voice_path, "voice_metadata.json")

        if os.path.isdir(voice_path) and os.path.exists(metadata_path):
            # Look for checkpoint folder
            checkpoint_dirs = [d for d in os.listdir(voice_path) if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                latest_checkpoint = sorted(checkpoint_dirs)[-1]
                model_path = os.path.join(voice_path, latest_checkpoint)
                voices[f"Trained: {voice_name}"] = {"type": "trained", "path": model_path}

    return voices


def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return models if models else ["qwen3.5:4b"]
    except:
        pass
    return ["qwen3.5:4b", "qwen3:4b", "mistral-nemo:latest"]


class VoiceChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Chat AI")
        self.root.geometry("850x750")
        self.root.resizable(True, True)

        # Chat state
        self.chat_thread = None
        self.chat_running = False
        self.output_queue = queue.Queue()
        self.voice_chat = None
        self.custom_voice_path = None

        # Training state
        self.train_thread = None
        self.training_running = False
        self.preprocessing_running = False
        self.train_output_queue = queue.Queue()
        self.training_pairs = []
        self.trained_voice_paths = {}

        # Studio state
        self.studio_audio_data = None
        self.studio_sample_rate = 24000
        self.studio_generating = False
        self.studio_recording = False
        self.studio_recorded_audio = None
        self.studio_output_queue = queue.Queue()
        self.studio_recorder = None
        self.studio_tts_engine_instance = None

        # Load saved config
        self.load_config()

        # Create UI
        self.create_widgets()

        # Start output polling
        self.poll_output()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_config(self):
        """Load saved configuration"""
        self.saved_config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    self.saved_config = json.load(f)
            except:
                pass

    def save_config(self):
        """Save current configuration"""
        config = {
            "persona": self.persona_var.get(),
            "system_prompt": self.prompt_text.get("1.0", tk.END).strip(),
            "input_mode": self.mode_var.get(),
            "whisper_model": self.whisper_var.get(),
            "ollama_model": self.ollama_var.get(),
            "voice": self.voice_var.get(),
            "fast_mode": self.fast_mode_var.get(),
            "streaming_mode": self.streaming_mode_var.get(),
            "tts_engine": self.tts_engine_var.get(),
            "qwen3_speaker": self.qwen3_speaker_var.get(),
            "omni_language": self.omni_lang_var.get(),
            "omni_voice_design": self.omni_design_var.get(),
            "omni_speed": self.omni_speed_var.get(),
            "studio_tts_engine": self.studio_engine_var.get(),
            "studio_voice": self.studio_voice_var.get(),
            "studio_qwen3_speaker": self.studio_qwen3_var.get(),
            "studio_omni_language": self.studio_omni_lang_var.get(),
            "studio_omni_design": self.studio_omni_design_var.get(),
            "studio_omni_speed": self.studio_omni_speed_var.get()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

    def create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabview (replaces ttk.Notebook)
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.pack(fill=tk.BOTH, expand=True)

        # Add tabs
        self.tabview.add("Chat")
        self.tabview.add("TTS Studio")
        self.tabview.add("Train Voice")

        # Get tab frames
        self.chat_frame = self.tabview.tab("Chat")
        self.studio_frame = self.tabview.tab("TTS Studio")
        self.train_frame = self.tabview.tab("Train Voice")

        # Build widgets for each tab
        self._create_chat_widgets()
        self._create_studio_widgets()
        self._create_train_widgets()

    def _create_chat_widgets(self):
        """Create widgets for the Chat tab"""
        # Top frame - Settings
        self.settings_frame_outer, self.settings_frame = create_section(self.chat_frame, "Settings")
        self.settings_frame_outer.pack(fill=tk.X, pady=2)

        # Row 1 - Persona and Models
        row1 = ctk.CTkFrame(self.settings_frame)
        row1.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(row1, text="Persona:").pack(side=tk.LEFT)
        self.persona_var = tk.StringVar(value=self.saved_config.get("persona", "Helpful Assistant"))
        persona_combo = ctk.CTkComboBox(row1, variable=self.persona_var,
                                      values=list(DEFAULT_PERSONAS.keys()), width=150,
                                      command=self.on_persona_change)
        persona_combo.pack(side=tk.LEFT, padx=5)

        ctk.CTkLabel(row1, text="LLM:").pack(side=tk.LEFT, padx=(10, 0))
        self.ollama_var = tk.StringVar(value=self.saved_config.get("ollama_model", "qwen3.5:4b"))
        self.ollama_combo = ctk.CTkComboBox(row1, variable=self.ollama_var,
                                          values=get_ollama_models(), width=160)
        self.ollama_combo.pack(side=tk.LEFT, padx=5)

        ctk.CTkLabel(row1, text="Whisper:").pack(side=tk.LEFT, padx=(10, 0))
        self.whisper_var = tk.StringVar(value=self.saved_config.get("whisper_model", "large-v3"))
        whisper_combo = ctk.CTkComboBox(row1, variable=self.whisper_var,
                                      values=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                                      width=120)
        whisper_combo.pack(side=tk.LEFT, padx=5)

        # Row 2 - Voice and Mode
        row2 = ctk.CTkFrame(self.settings_frame)
        row2.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(row2, text="Voice:").pack(side=tk.LEFT)
        # Default to first available voice (or first key in VOICE_OPTIONS)
        default_voice = list(VOICE_OPTIONS.keys())[0] if VOICE_OPTIONS else "Random (No Clone)"
        self.voice_var = tk.StringVar(value=self.saved_config.get("voice", default_voice))
        self.voice_combo = ctk.CTkComboBox(row2, variable=self.voice_var,
                                    values=list(VOICE_OPTIONS.keys()), width=200)
        self.voice_combo.pack(side=tk.LEFT, padx=5)
        self.voice_combo.configure(command=self.on_voice_change)

        ctk.CTkLabel(row2, text="Mode:").pack(side=tk.LEFT, padx=(10, 0))
        self.mode_var = tk.StringVar(value=self.saved_config.get("input_mode", "ptt"))
        ctk.CTkRadioButton(row2, text="Push-to-Talk", variable=self.mode_var,
                        value="ptt").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(row2, text="VAD", variable=self.mode_var,
                        value="vad").pack(side=tk.LEFT)

        # Fast mode checkbox
        self.fast_mode_var = tk.BooleanVar(value=self.saved_config.get("fast_mode", True))
        ctk.CTkCheckBox(row2, text="Fast Mode", variable=self.fast_mode_var).pack(side=tk.LEFT, padx=(20, 0))

        # Streaming mode checkbox (streams LLM->TTS for faster first response)
        self.streaming_mode_var = tk.BooleanVar(value=self.saved_config.get("streaming_mode", True))
        ctk.CTkCheckBox(row2, text="Streaming", variable=self.streaming_mode_var).pack(side=tk.LEFT, padx=(10, 0))

        # Row 3 - TTS Engine selection
        row3 = ctk.CTkFrame(self.settings_frame)
        row3.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(row3, text="TTS Engine:").pack(side=tk.LEFT)
        self.tts_engine_var = tk.StringVar(value=self.saved_config.get("tts_engine", "MOSS-TTS"))
        self.tts_engine_combo = ctk.CTkComboBox(row3, variable=self.tts_engine_var,
                                              values=TTS_ENGINE_OPTIONS, width=140)
        self.tts_engine_combo.pack(side=tk.LEFT, padx=5)
        self.tts_engine_combo.configure(command=self.on_tts_engine_change)

        # Qwen3-TTS Speaker selector (only visible when Qwen3-TTS selected)
        self.qwen3_speaker_label = ctk.CTkLabel(row3, text="Speaker:")
        self.qwen3_speaker_var = tk.StringVar(value=self.saved_config.get("qwen3_speaker", "serena"))
        self.qwen3_speaker_combo = ctk.CTkComboBox(row3, variable=self.qwen3_speaker_var,
                                                 values=QWEN3_SPEAKERS, width=120)

        # OmniVoice settings row (only visible when OmniVoice selected)
        self.omni_settings_row = ctk.CTkFrame(self.settings_frame)

        ctk.CTkLabel(self.omni_settings_row, text="Language:").pack(side=tk.LEFT)
        self.omni_lang_var = tk.StringVar(value=self.saved_config.get("omni_language", "en"))
        ctk.CTkComboBox(self.omni_settings_row, variable=self.omni_lang_var,
                     values=OMNIVOICE_LANGUAGES, width=80).pack(side=tk.LEFT, padx=5)

        ctk.CTkLabel(self.omni_settings_row, text="Voice Design:").pack(side=tk.LEFT, padx=(10, 0))
        self.omni_design_var = tk.StringVar(value=self.saved_config.get("omni_voice_design", ""))
        ctk.CTkEntry(self.omni_settings_row, textvariable=self.omni_design_var, width=200, state="disabled").pack(side=tk.LEFT, padx=5)

        ctk.CTkLabel(self.omni_settings_row, text="Speed:").pack(side=tk.LEFT, padx=(10, 0))
        self.omni_speed_var = tk.DoubleVar(value=float(self.saved_config.get("omni_speed", 1.0)))
        tk.Spinbox(self.omni_settings_row, from_=0.5, to=2.0, increment=0.1,
                    textvariable=self.omni_speed_var, width=5).pack(side=tk.LEFT, padx=2)

        # Show/hide engine-specific settings based on current engine selection
        self.on_tts_engine_change()

        # System prompt
        prompt_frame_outer, prompt_frame = create_section(self.settings_frame, "System Prompt")
        prompt_frame_outer.pack(fill=tk.X, pady=2)

        self.prompt_text = ctk.CTkTextbox(prompt_frame, wrap="word", height=80,
                                                      font=('Consolas', 11))
        self.prompt_text.pack(fill=tk.X)

        saved_prompt = self.saved_config.get("system_prompt", "")
        if saved_prompt:
            self.prompt_text.insert("1.0", saved_prompt)
        else:
            self.prompt_text.insert("1.0", DEFAULT_PERSONAS[self.persona_var.get()])

        # Voice Level Indicator
        level_frame = ctk.CTkFrame(self.chat_frame)
        level_frame.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(level_frame, text="Mic Level:").pack(side=tk.LEFT)
        self.level_canvas = tk.Canvas(level_frame, width=300, height=20, bg="#2b2b2b", highlightthickness=1)
        self.level_canvas.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.level_bar = self.level_canvas.create_rectangle(0, 0, 0, 20, fill="#4CAF50", outline="")

        # Conversation frame
        conv_frame_outer, conv_frame = create_section(self.chat_frame, "Conversation")
        conv_frame_outer.pack(fill=tk.BOTH, expand=True, pady=2)

        # Use tk.Text for colored tags support (CTkTextbox doesn't support tags)
        self.conv_text = tk.Text(conv_frame, wrap="word", state="disabled",
                                 font=('Consolas', 10), bg="#2b2b2b", fg="#dcdcdc",
                                 insertbackground="#dcdcdc", selectbackground="#4a6984",
                                 relief="flat", borderwidth=0, padx=8, pady=8)
        self.conv_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for colored text
        self.conv_text.tag_configure("user", foreground="#64B5F6", font=('Consolas', 10, 'bold'))
        self.conv_text.tag_configure("assistant", foreground="#81C784", font=('Consolas', 10, 'bold'))
        self.conv_text.tag_configure("system", foreground="#9E9E9E", font=('Consolas', 9, 'italic'))
        self.conv_text.tag_configure("error", foreground="#EF5350")

        # Bottom frame - Status and controls
        bottom_frame = ctk.CTkFrame(self.chat_frame)
        bottom_frame.pack(fill=tk.X, pady=5)

        # Status indicator
        status_frame = ctk.CTkFrame(bottom_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_indicator = tk.Canvas(status_frame, width=16, height=16)
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_circle = self.status_indicator.create_oval(2, 2, 14, 14, fill="gray")

        self.status_var = tk.StringVar(value="Ready - Configure settings and click Start")
        status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var, font=('Helvetica', 9))
        status_label.pack(side=tk.LEFT)

        # Buttons
        self.stop_btn = ctk.CTkButton(bottom_frame, text="Stop", command=self.stop_chat, state="disabled")
        self.stop_btn.pack(side=tk.RIGHT, padx=5)

        self.start_btn = ctk.CTkButton(bottom_frame, text="Start", command=self.start_chat)
        self.start_btn.pack(side=tk.RIGHT, padx=5)

        self.clear_btn = ctk.CTkButton(bottom_frame, text="Clear Log", command=self.clear_log)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)

        refresh_btn = ctk.CTkButton(bottom_frame, text="Refresh LLMs", command=self.refresh_llms)
        refresh_btn.pack(side=tk.RIGHT, padx=5)

        # Load trained voices into dropdown
        self.refresh_trained_voices()

    def _create_studio_widgets(self):
        """Create widgets for the TTS Studio tab"""
        container = ctk.CTkFrame(self.studio_frame, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True)

        # === Input Section ===
        input_frame_outer, input_frame = create_section(container, "Input")
        input_frame_outer.pack(fill=tk.X, pady=3)

        # Mode selector
        mode_row = ctk.CTkFrame(input_frame)
        mode_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(mode_row, text="Mode:").pack(side=tk.LEFT)
        self.studio_input_mode = tk.StringVar(value="text")
        ctk.CTkRadioButton(mode_row, text="Text", variable=self.studio_input_mode,
                        value="text", command=self.on_studio_input_mode_change).pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(mode_row, text="Microphone", variable=self.studio_input_mode,
                        value="mic", command=self.on_studio_input_mode_change).pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(mode_row, text="Audio File", variable=self.studio_input_mode,
                        value="file", command=self.on_studio_input_mode_change).pack(side=tk.LEFT, padx=5)

        # Text input frame
        self.studio_text_frame = ctk.CTkFrame(input_frame)
        self.studio_text_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        self.studio_text_input = ctk.CTkTextbox(self.studio_text_frame, wrap="word", height=120,
                                                            font=('Consolas', 11))
        self.studio_text_input.pack(fill=tk.BOTH, expand=True)
        self.studio_text_input.insert("1.0", "Hello there! This is a quick voice test. [laughter] Pretty cool, right? Let me know what you think about the quality of this voice.")

        # Mic input frame
        self.studio_mic_frame = ctk.CTkFrame(input_frame)
        mic_controls = ctk.CTkFrame(self.studio_mic_frame)
        mic_controls.pack(fill=tk.X, pady=2)
        self.studio_record_btn = ctk.CTkButton(mic_controls, text="Record", command=self.studio_toggle_record)
        self.studio_record_btn.pack(side=tk.LEFT, padx=5)
        self.studio_rec_duration_var = tk.StringVar(value="Duration: --")
        ctk.CTkLabel(mic_controls, textvariable=self.studio_rec_duration_var).pack(side=tk.LEFT, padx=10)
        ctk.CTkLabel(mic_controls, text="Mic Level:").pack(side=tk.LEFT, padx=(10, 0))
        self.studio_level_canvas = tk.Canvas(mic_controls, width=200, height=16, bg="#2b2b2b", highlightthickness=1)
        self.studio_level_canvas.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.studio_level_bar = self.studio_level_canvas.create_rectangle(0, 0, 0, 16, fill="#4CAF50", outline="")
        self.studio_transcribed_var = tk.StringVar(value="")
        ctk.CTkLabel(self.studio_mic_frame, textvariable=self.studio_transcribed_var,
                  font=('Consolas', 9, 'italic')).pack(anchor=tk.W, pady=2)

        # File input frame
        self.studio_file_frame = ctk.CTkFrame(input_frame)
        file_row = ctk.CTkFrame(self.studio_file_frame)
        file_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(file_row, text="Audio File:").pack(side=tk.LEFT)
        self.studio_file_var = tk.StringVar()
        ctk.CTkEntry(file_row, textvariable=self.studio_file_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(file_row, text="Browse...", command=self.studio_browse_file).pack(side=tk.LEFT)
        self.studio_file_info_var = tk.StringVar(value="")
        ctk.CTkLabel(self.studio_file_frame, textvariable=self.studio_file_info_var,
                  font=('Helvetica', 9, 'italic')).pack(anchor=tk.W, pady=2)

        # === TTS Settings Section ===
        settings_frame_outer, settings_frame = create_section(container, "TTS Settings")
        settings_frame_outer.pack(fill=tk.X, pady=3)

        settings_row1 = ctk.CTkFrame(settings_frame)
        settings_row1.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(settings_row1, text="Engine:").pack(side=tk.LEFT)
        self.studio_engine_var = tk.StringVar(value=self.saved_config.get("studio_tts_engine", "OmniVoice"))
        self.studio_engine_combo = ctk.CTkComboBox(settings_row1, variable=self.studio_engine_var,
                                                 values=TTS_ENGINE_OPTIONS, width=140)
        self.studio_engine_combo.pack(side=tk.LEFT, padx=5)
        self.studio_engine_combo.configure(command=self.on_studio_engine_change)

        ctk.CTkLabel(settings_row1, text="Voice:").pack(side=tk.LEFT, padx=(10, 0))
        default_voice = list(VOICE_OPTIONS.keys())[0] if VOICE_OPTIONS else "Random (No Clone)"
        self.studio_voice_var = tk.StringVar(value=self.saved_config.get("studio_voice", default_voice))
        self.studio_voice_combo = ctk.CTkComboBox(settings_row1, variable=self.studio_voice_var,
                                                values=list(VOICE_OPTIONS.keys()), width=200)
        self.studio_voice_combo.pack(side=tk.LEFT, padx=5)

        # Qwen3 speaker (conditional)
        self.studio_qwen3_label = ctk.CTkLabel(settings_row1, text="Speaker:")
        self.studio_qwen3_var = tk.StringVar(value=self.saved_config.get("studio_qwen3_speaker", "serena"))
        self.studio_qwen3_combo = ctk.CTkComboBox(settings_row1, variable=self.studio_qwen3_var,
                                                values=QWEN3_SPEAKERS, width=120)

        # OmniVoice settings (conditional)
        self.studio_omni_row = ctk.CTkFrame(settings_frame)
        ctk.CTkLabel(self.studio_omni_row, text="Language:").pack(side=tk.LEFT)
        self.studio_omni_lang_var = tk.StringVar(value=self.saved_config.get("studio_omni_language", "en"))
        ctk.CTkComboBox(self.studio_omni_row, variable=self.studio_omni_lang_var,
                     values=OMNIVOICE_LANGUAGES, width=80).pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(self.studio_omni_row, text="Voice Design:").pack(side=tk.LEFT, padx=(10, 0))
        self.studio_omni_design_var = tk.StringVar(value=self.saved_config.get("studio_omni_design", ""))
        ctk.CTkEntry(self.studio_omni_row, textvariable=self.studio_omni_design_var, width=200, state="disabled").pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(self.studio_omni_row, text="Speed:").pack(side=tk.LEFT, padx=(10, 0))
        self.studio_omni_speed_var = tk.DoubleVar(value=float(self.saved_config.get("studio_omni_speed", 1.0)))
        tk.Spinbox(self.studio_omni_row, from_=0.5, to=2.0, increment=0.1,
                    textvariable=self.studio_omni_speed_var, width=5).pack(side=tk.LEFT, padx=2)

        # Chunk size row
        chunk_row = ctk.CTkFrame(settings_frame)
        chunk_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(chunk_row, text="Chunk by:").pack(side=tk.LEFT)
        self.studio_chunk_mode = tk.StringVar(value="sentence")
        ctk.CTkRadioButton(chunk_row, text="Sentences", variable=self.studio_chunk_mode,
                          value="sentence").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(chunk_row, text="Max chars:", variable=self.studio_chunk_mode,
                          value="chars").pack(side=tk.LEFT, padx=5)
        self.studio_chunk_size_var = tk.IntVar(value=300)
        self.studio_chunk_size_spin = tk.Spinbox(chunk_row, from_=50, to=2000, increment=50,
                    textvariable=self.studio_chunk_size_var, width=6,
                    bg="#343638", fg="#dcdcdc", buttonbackground="#4a4a4a", relief="flat")
        self.studio_chunk_size_spin.pack(side=tk.LEFT, padx=3)

        self.on_studio_engine_change()

        # === Output Section ===
        output_frame_outer, output_frame = create_section(container, "Output")
        output_frame_outer.pack(fill=tk.BOTH, expand=True, pady=3)

        # Generate button + progress
        gen_row = ctk.CTkFrame(output_frame)
        gen_row.pack(fill=tk.X, pady=3)
        self.studio_generate_btn = ctk.CTkButton(gen_row, text="Generate", command=self.studio_generate)
        self.studio_generate_btn.pack(side=tk.LEFT, padx=5)
        self.studio_progress = ctk.CTkProgressBar(gen_row, mode="indeterminate")
        self.studio_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.studio_progress.set(0)

        # Audio info
        info_row = ctk.CTkFrame(output_frame)
        info_row.pack(fill=tk.X, pady=2)
        self.studio_duration_var = tk.StringVar(value="Duration: --")
        ctk.CTkLabel(info_row, textvariable=self.studio_duration_var, font=('Consolas', 9)).pack(side=tk.LEFT, padx=10)

        # Playback + save buttons
        playback_row = ctk.CTkFrame(output_frame)
        playback_row.pack(fill=tk.X, pady=3)
        self.studio_play_btn = ctk.CTkButton(playback_row, text="Play", command=self.studio_play, state="disabled")
        self.studio_play_btn.pack(side=tk.LEFT, padx=5)

        # Log output
        self.studio_log = tk.Text(output_frame, wrap="word", height=8,
                                   font=('Consolas', 9), state="disabled",
                                   bg="#2b2b2b", fg="#dcdcdc", relief="flat", borderwidth=0, padx=8, pady=8)
        self.studio_log.pack(fill=tk.BOTH, expand=True, pady=3)

        # Status
        self.studio_status_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(output_frame, textvariable=self.studio_status_var, font=('Helvetica', 9, 'bold')).pack(anchor=tk.W)

    # ============== Studio Methods ==============

    def on_studio_input_mode_change(self):
        mode = self.studio_input_mode.get()
        self.studio_text_frame.pack_forget()
        self.studio_mic_frame.pack_forget()
        self.studio_file_frame.pack_forget()
        if mode == "text":
            self.studio_text_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        elif mode == "mic":
            self.studio_mic_frame.pack(fill=tk.X, pady=2)
        elif mode == "file":
            self.studio_file_frame.pack(fill=tk.X, pady=2)

    def on_studio_engine_change(self, event=None):
        engine = self.studio_engine_var.get()
        # Qwen3 speaker
        if engine == "Qwen3-TTS":
            self.studio_qwen3_label.pack(side=tk.LEFT, padx=(15, 0))
            self.studio_qwen3_combo.pack(side=tk.LEFT, padx=5)
        else:
            self.studio_qwen3_label.pack_forget()
            self.studio_qwen3_combo.pack_forget()
        # OmniVoice row
        if engine == "OmniVoice":
            self.studio_omni_row.pack(fill=tk.X, pady=2)
        else:
            self.studio_omni_row.pack_forget()

    def studio_browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All Files", "*.*")]
        )
        if filepath:
            self.studio_file_var.set(filepath)
            try:
                import soundfile as sf
                info = sf.info(filepath)
                self.studio_file_info_var.set(f"{info.duration:.1f}s, {info.samplerate}Hz, {info.channels}ch")
            except Exception:
                self.studio_file_info_var.set(os.path.basename(filepath))

    def studio_toggle_record(self):
        if self.studio_recording:
            self.studio_stop_recording()
        else:
            self.studio_start_recording()

    def studio_start_recording(self):
        from audio_utils import AudioRecorder
        self.studio_recording = True
        self.studio_record_btn.configure(text="Stop Recording")
        self.studio_recorded_audio = None
        self.studio_rec_duration_var.set("Recording...")
        self.studio_transcribed_var.set("")

        if self.studio_recorder is None:
            self.studio_recorder = AudioRecorder()
        self.studio_recorder.level_callback = self.studio_on_level

        # Record in background thread
        def record_thread():
            audio = self.studio_recorder.record_ptt_toggle()
            self.studio_recorded_audio = audio
            if audio is not None and len(audio) > 0:
                duration = len(audio) / 16000
                self.root.after(0, lambda: self.studio_rec_duration_var.set(f"Duration: {duration:.1f}s"))
            self.root.after(0, lambda: self.studio_record_btn.configure(text="Record"))
            self.studio_recording = False

        threading.Thread(target=record_thread, daemon=True).start()

    def studio_stop_recording(self):
        self.studio_recording = False
        if self.studio_recorder:
            self.studio_recorder.recording = False

    def studio_on_level(self, level):
        width = int(level * 200)
        color = "#4CAF50" if level < 0.7 else "#FF9800" if level < 0.9 else "#F44336"
        self.studio_level_canvas.coords(self.studio_level_bar, 0, 0, width, 16)
        self.studio_level_canvas.itemconfig(self.studio_level_bar, fill=color)

    def studio_log_message(self, msg):
        self.studio_output_queue.put(("log", msg))

    def studio_generate(self):
        if self.studio_generating:
            return
        self.studio_generating = True
        self.studio_generate_btn.configure(state="disabled")
        self.studio_play_btn.configure(state="disabled")
        self.studio_progress.start()
        self.studio_status_var.set("Starting...")

        # Clear log
        self.studio_log.configure(state="normal")
        self.studio_log.delete("1.0", tk.END)
        self.studio_log.configure(state="disabled")

        threading.Thread(target=self._studio_generate_worker, daemon=True).start()
        self._poll_studio_output()

    def _studio_set_status(self, text):
        """Thread-safe status update"""
        self.root.after(0, lambda: self.studio_status_var.set(text))

    def _studio_generate_worker(self):
        try:
            import numpy as np
            import soundfile as sf
            from tts_engines import create_tts_engine

            mode = self.studio_input_mode.get()
            text = ""

            # Step 1: Get text
            if mode == "text":
                self._studio_set_status("Step 1/4: Reading input text...")
                text = self.studio_text_input.get("1.0", tk.END).strip()
                if not text:
                    self.studio_log_message("ERROR: Please enter text to synthesize")
                    return
                self.studio_log_message(f"Input text ({len(text)} chars): {text[:100]}...")

            elif mode == "mic":
                if self.studio_recorded_audio is None or len(self.studio_recorded_audio) < 8000:
                    self.studio_log_message("ERROR: No recording found. Please record audio first.")
                    return
                self._studio_set_status("Step 1/4: Transcribing recorded audio...")
                self.studio_log_message("Transcribing recorded audio with WhisperX...")
                text = self._studio_transcribe(self.studio_recorded_audio, 16000)
                if not text:
                    self.studio_log_message("ERROR: Could not transcribe audio")
                    return
                self.studio_log_message(f"Transcribed: {text}")
                self.root.after(0, lambda t=text: self.studio_transcribed_var.set(f"Transcribed: {t}"))

            elif mode == "file":
                filepath = self.studio_file_var.get()
                if not filepath or not os.path.exists(filepath):
                    self.studio_log_message("ERROR: Please select a valid audio file")
                    return
                self._studio_set_status("Step 1/4: Loading and transcribing audio file...")
                self.studio_log_message(f"Loading audio file: {os.path.basename(filepath)}")
                audio_data, sr = sf.read(filepath)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                if sr != 16000:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                self.studio_log_message("Transcribing audio file with WhisperX...")
                text = self._studio_transcribe(audio_data, 16000)
                if not text:
                    self.studio_log_message("ERROR: Could not transcribe audio")
                    return
                self.studio_log_message(f"Transcribed: {text}")

            # Step 2: Create TTS engine
            engine_name = self.studio_engine_var.get()
            self._studio_set_status(f"Step 2/4: Loading {engine_name} engine...")
            self.studio_log_message(f"Loading TTS engine: {engine_name}...")

            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            kwargs = {}
            if engine_name == "Qwen3-TTS":
                kwargs["speaker"] = self.studio_qwen3_var.get()
            elif engine_name == "OmniVoice":
                kwargs["language"] = self.studio_omni_lang_var.get()
                kwargs["voice_design"] = self.studio_omni_design_var.get()
                kwargs["speed"] = float(self.studio_omni_speed_var.get())
                voice_selection = self.studio_voice_var.get()
                vp = VOICE_OPTIONS.get(voice_selection)
                if vp and vp != "custom" and os.path.exists(str(vp)):
                    kwargs["voice_path"] = os.path.abspath(vp)

            # Reuse engine if same type already loaded
            if self.studio_tts_engine_instance and self.studio_tts_engine_instance.name == engine_name:
                engine = self.studio_tts_engine_instance
                self.studio_log_message(f"Reusing loaded {engine_name} engine")
            else:
                if self.studio_tts_engine_instance:
                    self.studio_tts_engine_instance.unload()
                engine = create_tts_engine(engine_name, device=device, dtype=dtype, **kwargs)
                self._studio_set_status(f"Step 2/4: Loading {engine_name} model to GPU...")
                engine.load()
                self.studio_tts_engine_instance = engine
            self.studio_log_message(f"{engine_name} ready on {device}")

            # Step 3: Get voice path for synthesis
            voice_selection = self.studio_voice_var.get()
            voice_path = None
            if not voice_selection.startswith("Random"):
                vp = VOICE_OPTIONS.get(voice_selection)
                if vp and vp != "custom" and os.path.exists(str(vp)):
                    voice_path = os.path.abspath(vp)

            if voice_path:
                self._studio_set_status("Step 2/4: Preparing voice reference transcript...")
                self._ensure_voice_transcript(voice_path)

            # Step 4: Synthesize (auto-chunk long texts)
            import time as _time
            sr = engine.get_sample_rate()

            chunk_mode = self.studio_chunk_mode.get()
            chunk_max = self.studio_chunk_size_var.get()
            chunks = self._split_text_for_tts(text, mode=chunk_mode, max_chars=chunk_max)
            total_chunks = len(chunks)
            self.studio_log_message(f"Synthesizing with {engine_name} ({total_chunks} chunk{'s' if total_chunks>1 else ''})...")

            audio_parts = []
            t_start = _time.time()
            for i, chunk in enumerate(chunks):
                self._studio_set_status(f"Step 3/4: Synthesizing chunk {i+1}/{total_chunks}...")
                if total_chunks > 1:
                    self.studio_log_message(f"  Chunk {i+1}/{total_chunks}: {chunk[:60]}...")
                chunk_audio = engine.synthesize(chunk, voice_path=voice_path)
                if len(chunk_audio) > 0:
                    audio_parts.append(chunk_audio)
                    chunk_dur = len(chunk_audio) / sr
                    elapsed = _time.time() - t_start
                    self.studio_log_message(f"  Chunk {i+1} done: {chunk_dur:.1f}s audio ({elapsed:.1f}s elapsed)")
            t_gen = _time.time() - t_start

            # Combine all chunks
            self._studio_set_status("Step 4/4: Combining audio and saving...")
            if audio_parts:
                silence = np.zeros(int(sr * 0.15), dtype=np.float32)
                combined = []
                for j, part in enumerate(audio_parts):
                    combined.append(part)
                    if j < len(audio_parts) - 1:
                        combined.append(silence)
                audio = np.concatenate(combined)
            else:
                audio = np.array([], dtype=np.float32)

            self.studio_audio_data = audio
            self.studio_sample_rate = sr

            duration = len(audio) / sr if len(audio) > 0 else 0
            self.studio_log_message(f"Generated {duration:.1f}s of audio in {t_gen:.1f}s (RTF: {t_gen/duration:.2f}x)" if duration > 0 else "No audio generated")

            # Auto-save to Output folder
            if len(audio) > 0:
                self._studio_auto_save(audio, sr)

            self.studio_output_queue.put(("done", f"Duration: {duration:.1f}s at {sr}Hz"))

        except Exception as e:
            self.studio_log_message(f"ERROR: {str(e)}")
            import traceback
            self.studio_log_message(traceback.format_exc())
        finally:
            self.studio_generating = False
            self.root.after(0, self._studio_on_generate_done)

    def _split_text_for_tts(self, text, mode="sentence", max_chars=300):
        """Split long text into chunks for TTS processing.
        mode='sentence': split only at sentence boundaries (. ! ?)
        mode='chars': split at sentence boundaries but enforce max_chars limit
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        if mode == "sentence":
            # Each sentence is its own chunk
            return sentences

        # mode == "chars": group sentences up to max_chars
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_chars:
                current = (current + " " + sentence).strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                # If single sentence is too long, split at commas
                if len(sentence) > max_chars:
                    parts = sentence.split(', ')
                    sub = ""
                    for part in parts:
                        if len(sub) + len(part) + 2 <= max_chars:
                            sub = (sub + ", " + part).strip(', ') if sub else part
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = part
                    current = sub
                else:
                    current = sentence

        if current:
            chunks.append(current)
        return chunks if chunks else [text]

    def _studio_auto_save(self, audio, sr):
        """Auto-save generated audio to Output folder as MP3"""
        try:
            import numpy as np
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, OUTPUT_DIR)
            os.makedirs(output_dir, exist_ok=True)

            filename = self._generate_output_filename("mp3")
            filepath = os.path.join(output_dir, filename)

            from pydub import AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            segment = AudioSegment(
                data=audio_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1
            )
            segment.export(filepath, format="mp3", bitrate="192k")
            self._studio_last_saved_path = filepath
            self.studio_log_message(f"Auto-saved: {filename}")
        except Exception as e:
            self.studio_log_message(f"WARNING: Auto-save failed: {e}")

    def _ensure_voice_transcript(self, voice_path):
        """Ensure a .txt transcript exists next to the voice sample. Transcribe if missing."""
        base = os.path.splitext(voice_path)[0]
        txt_path = base + ".txt"
        if os.path.exists(txt_path):
            return  # Already has transcript

        self.studio_log_message(f"No transcript found for {os.path.basename(voice_path)}, transcribing...")
        try:
            import soundfile as sf
            import numpy as np
            audio_data, sr = sf.read(voice_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            if sr != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

            # Use medium model for higher quality transcription
            text = self._studio_transcribe(audio_data, 16000, whisper_model="medium")
            if text:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.studio_log_message(f"Transcript saved: {os.path.basename(txt_path)}")
                self.studio_log_message(f"  Text: {text[:100]}...")
            else:
                self.studio_log_message("WARNING: Could not transcribe voice sample")
        except Exception as e:
            self.studio_log_message(f"WARNING: Transcript generation failed: {e}")

    def _studio_transcribe(self, audio_np, sample_rate, whisper_model="base"):
        """Standalone WhisperX transcription for Studio tab"""
        import numpy as np
        import soundfile as sf
        import tempfile
        import warnings
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, sample_rate, subtype='PCM_16')
            temp_path = f.name

        try:
            import whisperx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = whisperx.load_model(whisper_model, device, compute_type="float16")
                result = model.transcribe(temp_path, batch_size=16, language="en")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not result or "segments" not in result or not result["segments"]:
                return ""
            return " ".join([seg["text"] for seg in result["segments"]]).strip()
        except Exception as e:
            self.studio_log_message(f"Transcription error: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def _studio_on_generate_done(self):
        self.studio_progress.stop()
        self.studio_generate_btn.configure(state="normal")
        if self.studio_audio_data is not None and len(self.studio_audio_data) > 0:
            self.studio_play_btn.configure(state="normal")
            duration = len(self.studio_audio_data) / self.studio_sample_rate
            self.studio_duration_var.set(f"Duration: {duration:.1f}s at {self.studio_sample_rate}Hz")
            self.studio_status_var.set(f"Done - {duration:.1f}s (auto-saved to Output)")
        else:
            self.studio_status_var.set("Generation failed")

    def _poll_studio_output(self):
        try:
            while True:
                msg_type, data = self.studio_output_queue.get_nowait()
                if msg_type == "log":
                    self.studio_log.configure(state="normal")
                    self.studio_log.insert(tk.END, data + "\n")
                    self.studio_log.see(tk.END)
                    self.studio_log.configure(state="disabled")
                elif msg_type == "done":
                    self.studio_duration_var.set(data)
        except queue.Empty:
            pass
        if self.studio_generating:
            self.root.after(100, self._poll_studio_output)

    def studio_play(self):
        """Open the last saved MP3 in the system's default audio player"""
        if not hasattr(self, '_studio_last_saved_path') or not self._studio_last_saved_path:
            return
        if not os.path.exists(self._studio_last_saved_path):
            return
        os.startfile(self._studio_last_saved_path)

    def _generate_output_filename(self, ext="mp3"):
        """Generate output filename: VoiceName_Engine_YYYYMMDD_HHMMSS.ext"""
        from datetime import datetime
        voice = self.studio_voice_var.get()
        # Clean voice name for filename
        voice_clean = voice.replace("Sample: ", "").replace("Trained: ", "")
        voice_clean = re.sub(r'[^\w\-]', '_', voice_clean).strip('_')
        if not voice_clean or voice_clean.startswith("Random"):
            voice_clean = "DefaultVoice"
        engine = self.studio_engine_var.get().replace("-", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{voice_clean}_{engine}_{timestamp}.{ext}"

    def studio_save_mp3(self):
        if self.studio_audio_data is None:
            return

        # Ensure Output directory exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)

        # Generate default filename
        default_name = self._generate_output_filename("mp3")

        filepath = filedialog.asksaveasfilename(
            initialdir=output_dir,
            initialfile=default_name,
            defaultextension=".mp3",
            filetypes=[("MP3 Audio", "*.mp3"), ("WAV Audio", "*.wav"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        try:
            import numpy as np
            if filepath.lower().endswith(".wav"):
                import soundfile as sf
                sf.write(filepath, self.studio_audio_data, self.studio_sample_rate, subtype='PCM_16')
            else:
                from pydub import AudioSegment
                audio_int16 = (self.studio_audio_data * 32767).astype(np.int16)
                segment = AudioSegment(
                    data=audio_int16.tobytes(),
                    sample_width=2,
                    frame_rate=self.studio_sample_rate,
                    channels=1
                )
                segment.export(filepath, format="mp3", bitrate="192k")
            self.studio_status_var.set(f"Saved: {os.path.basename(filepath)}")
            self.studio_log_message(f"Saved to: {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file: {e}")

    def _create_train_widgets(self):
        """Create widgets for the Train Voice tab"""
        # Container frame
        train_container = ctk.CTkFrame(self.train_frame, fg_color="transparent")
        train_container.pack(fill=tk.BOTH, expand=True)

        # --- Data Source Section ---
        data_frame_outer, data_frame = create_section(train_container, "Training Data")
        data_frame_outer.pack(fill=tk.X, pady=5)

        # Data source mode selector
        mode_row = ctk.CTkFrame(data_frame)
        mode_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(mode_row, text="Source:").pack(side=tk.LEFT)
        self.data_source_mode = tk.StringVar(value="folder")
        ctk.CTkRadioButton(mode_row, text="Folder with pairs", variable=self.data_source_mode,
                        value="folder", command=self.on_data_source_change).pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(mode_row, text="Single audio file (auto-transcribe)", variable=self.data_source_mode,
                        value="single", command=self.on_data_source_change).pack(side=tk.LEFT, padx=5)

        # --- Folder mode widgets ---
        self.folder_widgets_frame = ctk.CTkFrame(data_frame)
        self.folder_widgets_frame.pack(fill=tk.X, pady=2)

        folder_row = ctk.CTkFrame(self.folder_widgets_frame)
        folder_row.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(folder_row, text="Audio Folder:").pack(side=tk.LEFT)
        self.train_folder_var = tk.StringVar()
        self.train_folder_entry = ctk.CTkEntry(folder_row, textvariable=self.train_folder_var, width=50)
        self.train_folder_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(folder_row, text="Browse...", command=self.browse_train_folder).pack(side=tk.LEFT)

        self.folder_format_label = ctk.CTkLabel(self.folder_widgets_frame,
                                              text="Expected format: audio1.wav + audio1.txt, audio2.mp3 + audio2.txt, etc.",
                                              font=('Helvetica', 8), text_color="gray")
        self.folder_format_label.pack(anchor=tk.W)

        # --- Single file mode widgets ---
        self.single_file_frame = ctk.CTkFrame(data_frame)
        # Initially hidden, shown when single file mode selected

        single_file_row = ctk.CTkFrame(self.single_file_frame)
        single_file_row.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(single_file_row, text="Audio File:").pack(side=tk.LEFT)
        self.single_audio_var = tk.StringVar()
        self.single_audio_entry = ctk.CTkEntry(single_file_row, textvariable=self.single_audio_var, width=50)
        self.single_audio_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(single_file_row, text="Browse...", command=self.browse_single_audio).pack(side=tk.LEFT)

        # Preprocessing options
        preprocess_row = ctk.CTkFrame(self.single_file_frame)
        preprocess_row.pack(fill=tk.X, pady=2)

        ctk.CTkLabel(preprocess_row, text="Whisper Model:").pack(side=tk.LEFT)
        self.preprocess_whisper_var = tk.StringVar(value="base")
        ctk.CTkComboBox(preprocess_row, variable=self.preprocess_whisper_var,
                     values=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                     width=120).pack(side=tk.LEFT, padx=5)

        ctk.CTkLabel(preprocess_row, text="Min segment (s):").pack(side=tk.LEFT, padx=(10, 0))
        self.min_segment_var = tk.DoubleVar(value=2.0)
        tk.Spinbox(preprocess_row, from_=0.5, to=10.0, increment=0.5,
                    textvariable=self.min_segment_var, width=6).pack(side=tk.LEFT, padx=5)

        ctk.CTkLabel(preprocess_row, text="Max segment (s):").pack(side=tk.LEFT, padx=(10, 0))
        self.max_segment_var = tk.DoubleVar(value=15.0)
        tk.Spinbox(preprocess_row, from_=5.0, to=30.0, increment=1.0,
                    textvariable=self.max_segment_var, width=6).pack(side=tk.LEFT, padx=5)

        self.preprocess_btn = ctk.CTkButton(self.single_file_frame, text="Preprocess with WhisperX",
                                          command=self.start_preprocessing)
        self.preprocess_btn.pack(anchor=tk.W, pady=5)

        single_info = ctk.CTkLabel(self.single_file_frame,
                                 text="Preprocesses long audio: transcribes with WhisperX, splits into segments, creates training pairs",
                                 font=('Helvetica', 8), text_color="gray")
        single_info.pack(anchor=tk.W)

        # Folder info label (shared between modes)
        self.folder_info_var = tk.StringVar(value="Select a folder containing audio files with matching .txt transcripts")
        ctk.CTkLabel(data_frame, textvariable=self.folder_info_var, font=('Helvetica', 9, 'italic')).pack(anchor=tk.W, pady=2)

        # --- Training Settings Section ---
        settings_frame_outer, settings_frame = create_section(train_container, "Training Settings")
        settings_frame_outer.pack(fill=tk.X, pady=5)

        # Voice name
        name_row = ctk.CTkFrame(settings_frame)
        name_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(name_row, text="Voice Name:").pack(side=tk.LEFT)
        self.voice_name_var = tk.StringVar()
        ctk.CTkEntry(name_row, textvariable=self.voice_name_var, width=250).pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(name_row, text="(used to identify trained voice)", font=('Helvetica', 8), text_color="gray").pack(side=tk.LEFT)

        # Sample count selector
        samples_row = ctk.CTkFrame(settings_frame)
        samples_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(samples_row, text="Samples to use:").pack(side=tk.LEFT)
        self.sample_count_var = tk.IntVar(value=10)
        self.sample_spin = tk.Spinbox(samples_row, from_=1, to=1000, textvariable=self.sample_count_var, width=10)
        self.sample_spin.pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(samples_row, text="(more samples = better quality, longer training)",
                  font=('Helvetica', 9, 'italic')).pack(side=tk.LEFT)

        # Epochs selector
        epochs_row = ctk.CTkFrame(settings_frame)
        epochs_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(epochs_row, text="Training Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=3)
        tk.Spinbox(epochs_row, from_=1, to=10, textvariable=self.epochs_var, width=10).pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(epochs_row, text="(more epochs = better fit, risk of overfitting)",
                  font=('Helvetica', 9, 'italic')).pack(side=tk.LEFT)

        # Model info (fixed)
        model_row = ctk.CTkFrame(settings_frame)
        model_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(model_row, text="Model: MossTTSLocal 1.7B", font=('Helvetica', 9)).pack(side=tk.LEFT)
        ctk.CTkLabel(model_row, text="(optimized for RTX 4090)", font=('Helvetica', 8), text_color="gray").pack(side=tk.LEFT, padx=5)

        # --- Progress Section ---
        progress_frame_outer, progress_frame = create_section(train_container, "Training Progress")
        progress_frame_outer.pack(fill=tk.BOTH, expand=True, pady=5)

        # Progress bar
        self.train_progress = ctk.CTkProgressBar(progress_frame)
        self.train_progress.pack(fill=tk.X, pady=5)
        self.train_progress.set(0)

        # Status label
        self.train_status_var = tk.StringVar(value="Ready to train")
        ctk.CTkLabel(progress_frame, textvariable=self.train_status_var, font=('Helvetica', 9, 'bold')).pack(anchor=tk.W)

        # Log output
        self.train_log = tk.Text(progress_frame, wrap="word", height=15,
                                  font=('Consolas', 9), state="disabled",
                                  bg="#2b2b2b", fg="#dcdcdc", relief="flat", borderwidth=0, padx=8, pady=8)
        self.train_log.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Control Buttons ---
        btn_frame = ctk.CTkFrame(train_container)
        btn_frame.pack(fill=tk.X, pady=10)

        self.scan_folder_btn = ctk.CTkButton(btn_frame, text="Scan Folder", command=self.scan_training_folder)
        self.scan_folder_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_train_btn = ctk.CTkButton(btn_frame, text="Cancel", command=self.cancel_training, state="disabled")
        self.cancel_train_btn.pack(side=tk.RIGHT, padx=5)

        self.start_train_btn = ctk.CTkButton(btn_frame, text="Start Training", command=self.start_training)
        self.start_train_btn.pack(side=tk.RIGHT, padx=5)

    def on_persona_change(self, event=None):
        """Update prompt when persona changes"""
        persona = self.persona_var.get()
        if persona != "Custom":
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", DEFAULT_PERSONAS[persona])

    def on_voice_change(self, event=None):
        """Handle voice selection change"""
        voice = self.voice_var.get()
        if voice == "Custom Audio File...":
            filepath = filedialog.askopenfilename(
                title="Select Voice Reference Audio",
                filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg"), ("All Files", "*.*")]
            )
            if filepath:
                self.custom_voice_path = filepath
                self.voice_var.set(f"Custom: {os.path.basename(filepath)}")
            else:
                self.voice_var.set("Default (No Clone)")

    def on_tts_engine_change(self, event=None):
        """Handle TTS engine selection change - show/hide engine-specific settings"""
        engine = self.tts_engine_var.get()

        # Qwen3-TTS speaker controls
        if engine == "Qwen3-TTS":
            self.qwen3_speaker_label.pack(side=tk.LEFT, padx=(15, 0))
            self.qwen3_speaker_combo.pack(side=tk.LEFT, padx=5)
        else:
            self.qwen3_speaker_label.pack_forget()
            self.qwen3_speaker_combo.pack_forget()

        # OmniVoice settings row
        if engine == "OmniVoice":
            self.omni_settings_row.pack(fill=tk.X, pady=2)
        else:
            self.omni_settings_row.pack_forget()

    def refresh_llms(self):
        """Refresh the list of available Ollama models"""
        models = get_ollama_models()
        self.ollama_combo.configure(values=models)
        self.set_status(f"Found {len(models)} models", "green")

    def set_status(self, text, color="gray"):
        """Update status indicator and text"""
        self.status_var.set(text)
        self.status_indicator.itemconfig(self.status_circle, fill=color)

    def update_level(self, level):
        """Update the voice level indicator (0.0 to 1.0)"""
        width = int(level * 300)
        color = "#4CAF50" if level < 0.7 else "#FF9800" if level < 0.9 else "#F44336"
        self.level_canvas.coords(self.level_bar, 0, 0, width, 20)
        self.level_canvas.itemconfig(self.level_bar, fill=color)

    def append_conv(self, text, tag=None):
        """Append text to conversation log"""
        self.conv_text.configure(state="normal")
        if tag:
            self.conv_text.insert(tk.END, text, tag)
        else:
            self.conv_text.insert(tk.END, text)
        self.conv_text.see(tk.END)
        self.conv_text.configure(state="disabled")

    def clear_log(self):
        """Clear conversation log"""
        self.conv_text.configure(state="normal")
        self.conv_text.delete("1.0", tk.END)
        self.conv_text.configure(state="disabled")

    def poll_output(self):
        """Poll for output from the chat thread"""
        try:
            while True:
                tag, text = self.output_queue.get_nowait()
                if tag == "status":
                    color = "green" if "Running" in text else "orange" if "Loading" in text or "Processing" in text or "Transcribing" in text or "Thinking" in text or "Speaking" in text else "gray"
                    self.set_status(text, color)
                elif tag == "level":
                    self.update_level(float(text))
                elif tag == "user":
                    self.append_conv(f"\nYou: ", "user")
                    self.append_conv(f"{text}\n")
                elif tag == "assistant":
                    self.append_conv(f"\nAssistant: ", "assistant")
                    self.append_conv(f"{text}\n")
                elif tag == "system":
                    self.append_conv(f"[{text}]\n", "system")
                elif tag == "error":
                    self.append_conv(f"ERROR: {text}\n", "error")
                else:
                    self.append_conv(f"{text}\n", "system")
        except queue.Empty:
            pass

        # Schedule next poll
        self.root.after(50, self.poll_output)

    def start_chat(self):
        """Start the voice chat in a background thread"""
        if self.chat_running:
            return

        # Save settings
        self.save_config()

        # Update UI state
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.prompt_text.configure(state="disabled")
        self.set_status("Starting...", "orange")

        # Get settings
        system_prompt = self.prompt_text.get("1.0", tk.END).strip()
        input_mode = self.mode_var.get()
        whisper_model = self.whisper_var.get()
        ollama_model = self.ollama_var.get()
        fast_mode = self.fast_mode_var.get()

        # Get voice path
        voice_selection = self.voice_var.get()
        trained_model_path = None
        if voice_selection.startswith("Custom:"):
            voice_path = self.custom_voice_path
        elif voice_selection.startswith("Trained:"):
            # Use trained model instead of reference audio
            voice_path = None
            trained_model_path = self.trained_voice_paths.get(voice_selection, {}).get("path")
        else:
            voice_path = VOICE_OPTIONS.get(voice_selection)

        # Get TTS engine settings
        tts_engine = self.tts_engine_var.get()
        qwen3_speaker = self.qwen3_speaker_var.get()
        omni_language = self.omni_lang_var.get()
        omni_voice_design = self.omni_design_var.get()
        omni_speed = float(self.omni_speed_var.get() or 1.0)

        # Start chat thread
        self.chat_running = True
        self.chat_thread = threading.Thread(
            target=self.run_chat_loop,
            args=(system_prompt, input_mode, whisper_model, ollama_model, voice_path, fast_mode, trained_model_path, tts_engine, qwen3_speaker, omni_language, omni_voice_design, omni_speed),
            daemon=True
        )
        self.chat_thread.start()

    def stop_chat(self):
        """Stop the voice chat"""
        self.chat_running = False
        if self.voice_chat:
            self.voice_chat.running = False

        self.set_status("Stopping...", "orange")
        self.update_level(0)

        # Update UI state
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.prompt_text.configure(state="normal")

        self.output_queue.put(("system", "Chat stopped by user"))
        self.set_status("Stopped - Click Start to begin again", "gray")

    def run_chat_loop(self, system_prompt, input_mode, whisper_model, ollama_model, voice_path, fast_mode, trained_model_path=None, tts_engine="MOSS-TTS", qwen3_speaker="serena", omni_language="en", omni_voice_design="", omni_speed=1.0):
        """Run the voice chat loop in a background thread"""
        try:
            import torch
            import numpy as np

            # Add paths
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MOSS-TTS'))
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MOSS-TTS', 'moss_tts_realtime'))

            import config
            config.SYSTEM_PROMPT = system_prompt
            config.DEFAULT_INPUT_MODE = input_mode
            config.WHISPER_MODEL = whisper_model

            self.output_queue.put(("status", "Loading models..."))
            self.output_queue.put(("system", f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"))
            self.output_queue.put(("system", f"LLM: {ollama_model}"))
            self.output_queue.put(("system", f"TTS Engine: {tts_engine}"))
            if tts_engine == "Qwen3-TTS":
                self.output_queue.put(("system", f"Qwen3 Speaker: {qwen3_speaker}"))
            elif tts_engine == "OmniVoice":
                self.output_queue.put(("system", f"OmniVoice Language: {omni_language}, Speed: {omni_speed}"))
                if omni_voice_design:
                    self.output_queue.put(("system", f"Voice Design: {omni_voice_design}"))
            if trained_model_path:
                self.output_queue.put(("system", f"Using trained voice model: {os.path.basename(trained_model_path)}"))

            # Create chat handler with streaming mode enabled by default
            streaming_mode = getattr(self, 'streaming_mode_var', None)
            streaming_enabled = streaming_mode.get() if streaming_mode else True

            chat = SimplifiedVoiceChat(
                output_queue=self.output_queue,
                system_prompt=system_prompt,
                whisper_model=whisper_model,
                input_mode=input_mode,
                ollama_model=ollama_model,
                voice_path=voice_path,
                fast_mode=fast_mode,
                trained_model_path=trained_model_path,
                streaming_mode=streaming_enabled,
                tts_engine=tts_engine,
                qwen3_speaker=qwen3_speaker,
                omni_language=omni_language,
                omni_voice_design=omni_voice_design,
                omni_speed=omni_speed
            )
            self.voice_chat = chat

            # Preload all models into VRAM
            chat.preload_all_models()

            self.output_queue.put(("system", f"Input mode: {input_mode.upper()}"))
            self.output_queue.put(("system", f"Fast mode: {'ON' if fast_mode else 'OFF'}"))
            self.output_queue.put(("system", f"Streaming mode: {'ON' if streaming_enabled else 'OFF'}"))
            self.output_queue.put(("system", "Hold SPACE to speak, TAB to toggle modes"))
            self.output_queue.put(("status", "Running - Hold SPACE to speak"))

            # Main loop - use streaming or batch mode
            while self.chat_running:
                try:
                    if chat.streaming_mode:
                        chat.process_turn_streaming()
                    else:
                        chat.process_turn()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.output_queue.put(("error", str(e)))
                    import traceback
                    traceback.print_exc()

            chat.cleanup()

        except Exception as e:
            self.output_queue.put(("error", f"Failed to start: {e}"))
            import traceback
            traceback.print_exc()

        finally:
            self.chat_running = False
            self.root.after(0, self.on_chat_stopped)

    def on_chat_stopped(self):
        """Called when chat thread ends"""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.prompt_text.configure(state="normal")
        self.set_status("Stopped - Click Start to begin again", "gray")
        self.update_level(0)

    # ============== Training Methods ==============

    def browse_train_folder(self):
        """Open folder picker for training data"""
        folder = filedialog.askdirectory(title="Select folder with audio files and transcripts")
        if folder:
            self.train_folder_var.set(folder)
            self.scan_training_folder()

    def on_data_source_change(self):
        """Toggle between folder and single file modes"""
        mode = self.data_source_mode.get()
        if mode == "folder":
            self.single_file_frame.pack_forget()
            self.folder_widgets_frame.pack(fill=tk.X, pady=2)
            self.folder_info_var.set("Select a folder containing audio files with matching .txt transcripts")
        else:
            self.folder_widgets_frame.pack_forget()
            self.single_file_frame.pack(fill=tk.X, pady=2)
            self.folder_info_var.set("Select a long audio file to preprocess into training segments")

    def browse_single_audio(self):
        """Open file picker for single audio file"""
        filepath = filedialog.askopenfilename(
            title="Select audio file to preprocess",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All Files", "*.*")]
        )
        if filepath:
            self.single_audio_var.set(filepath)
            # Get audio duration for info
            try:
                import soundfile as sf
                info = sf.info(filepath)
                duration_mins = info.duration / 60
                self.folder_info_var.set(f"Selected: {os.path.basename(filepath)} ({duration_mins:.1f} minutes)")
            except Exception as e:
                self.folder_info_var.set(f"Selected: {os.path.basename(filepath)}")

    def start_preprocessing(self):
        """Start preprocessing in a background thread"""
        audio_path = self.single_audio_var.get()
        if not audio_path or not os.path.exists(audio_path):
            messagebox.showerror("Error", "Please select a valid audio file")
            return

        # Disable button during processing
        self.preprocess_btn.configure(state="disabled")
        self.train_progress.set(0)

        # Clear log
        self.train_log.configure(state="normal")
        self.train_log.delete("1.0", tk.END)
        self.train_log.configure(state="disabled")

        # Start preprocessing thread
        self.preprocessing_running = True
        preprocess_thread = threading.Thread(
            target=self.run_preprocessing,
            args=(audio_path,),
            daemon=True
        )
        preprocess_thread.start()

        # Start polling for output
        self.poll_training_output()

    def run_preprocessing(self, audio_path: str):
        """Run WhisperX preprocessing to split audio into training segments"""
        try:
            import torch
            import soundfile as sf
            import numpy as np

            self.train_log_message("=" * 50)
            self.train_log_message("PREPROCESSING: Starting WhisperX transcription...")
            self.train_log_message("=" * 50)
            self.train_status_var.set("Preprocessing: Loading audio...")
            self.update_train_progress(5)

            # Load audio
            audio_data, sample_rate = sf.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert stereo to mono

            duration_secs = len(audio_data) / sample_rate
            self.train_log_message(f"Audio loaded: {duration_secs/60:.1f} minutes at {sample_rate}Hz")

            # Resample to 16kHz if needed (WhisperX requirement)
            if sample_rate != 16000:
                self.train_log_message(f"Resampling from {sample_rate}Hz to 16000Hz...")
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            self.update_train_progress(10)
            self.train_status_var.set("Preprocessing: Loading WhisperX...")

            # Load WhisperX
            import whisperx
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = self.preprocess_whisper_var.get()
            self.train_log_message(f"Loading WhisperX model '{whisper_model}' on {device}...")

            # Set up cache directory for model downloads
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.train_log_message(f"Model cache directory: {cache_dir}")

            try:
                model = whisperx.load_model(
                    whisper_model,
                    device,
                    compute_type="float16",
                    download_root=cache_dir
                )
            except Exception as e:
                self.train_log_message(f"Error loading model: {e}")
                self.train_log_message("If download fails, try a smaller model (base/small) or check your internet connection.")
                self.train_log_message("For large-v3, you may need to accept the license at: https://huggingface.co/openai/whisper-large-v3")
                raise
            self.update_train_progress(20)

            # Save temp file for WhisperX (it needs a file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, sample_rate, subtype='PCM_16')
                temp_audio_path = f.name

            self.train_status_var.set("Preprocessing: Transcribing with WhisperX...")
            self.train_log_message("Running transcription (this may take a while for long audio)...")

            # Transcribe
            result = model.transcribe(temp_audio_path, batch_size=16, language="en")
            self.update_train_progress(50)

            # Get word-level timestamps with alignment
            self.train_log_message("Aligning words for precise timestamps...")
            self.train_status_var.set("Preprocessing: Aligning timestamps...")

            try:
                model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
                result = whisperx.align(result["segments"], model_a, metadata, temp_audio_path, device)
                del model_a  # Free memory
            except Exception as e:
                self.train_log_message(f"Warning: Alignment failed ({e}), using segment-level timestamps")

            # Clean up WhisperX model to free VRAM
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.update_train_progress(60)

            # Clean up temp file
            try:
                os.unlink(temp_audio_path)
            except:
                pass

            segments = result.get("segments", [])
            self.train_log_message(f"Found {len(segments)} segments from transcription")

            if not segments:
                self.train_log_message("ERROR: No segments found in transcription!")
                self.train_status_var.set("Error: No speech detected")
                return

            # Create output folder
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_folder = os.path.join(os.path.dirname(audio_path), f"{base_name}_segments")
            os.makedirs(output_folder, exist_ok=True)
            self.train_log_message(f"Output folder: {output_folder}")

            self.train_status_var.set("Preprocessing: Splitting audio segments...")
            self.update_train_progress(65)

            # Reload original audio at original sample rate for better quality output
            audio_data_orig, sr_orig = sf.read(audio_path)
            if len(audio_data_orig.shape) > 1:
                audio_data_orig = audio_data_orig.mean(axis=1)

            # Process segments - merge short ones, split long ones
            min_duration = self.min_segment_var.get()
            max_duration = self.max_segment_var.get()

            processed_segments = []
            current_segment = None

            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "").strip()

                if not text:
                    continue

                duration = end - start

                if current_segment is None:
                    current_segment = {"start": start, "end": end, "text": text}
                elif current_segment["end"] - current_segment["start"] + duration < max_duration:
                    # Merge with current segment
                    current_segment["end"] = end
                    current_segment["text"] += " " + text
                else:
                    # Save current and start new
                    if current_segment["end"] - current_segment["start"] >= min_duration:
                        processed_segments.append(current_segment)
                    current_segment = {"start": start, "end": end, "text": text}

            # Don't forget last segment
            if current_segment and current_segment["end"] - current_segment["start"] >= min_duration:
                processed_segments.append(current_segment)

            self.train_log_message(f"Processed into {len(processed_segments)} training segments")
            self.update_train_progress(70)

            # Save segments as audio + text files
            saved_count = 0
            for i, seg in enumerate(processed_segments):
                if not self.preprocessing_running:
                    self.train_log_message("Preprocessing cancelled.")
                    return

                start_sample = int(seg["start"] * sr_orig)
                end_sample = int(seg["end"] * sr_orig)
                segment_audio = audio_data_orig[start_sample:end_sample]

                # Generate filenames
                segment_name = f"segment_{i+1:04d}"
                audio_out_path = os.path.join(output_folder, f"{segment_name}.wav")
                text_out_path = os.path.join(output_folder, f"{segment_name}.txt")

                # Save audio as high-quality PCM WAV (16-bit, lossless)
                sf.write(audio_out_path, segment_audio, sr_orig, subtype='PCM_16')

                # Save transcript
                with open(text_out_path, 'w', encoding='utf-8') as f:
                    f.write(seg["text"])

                saved_count += 1

                # Update progress
                progress = 70 + (25 * (i + 1) / len(processed_segments))
                self.update_train_progress(progress)

                if (i + 1) % 10 == 0:
                    self.train_log_message(f"  Saved {i+1}/{len(processed_segments)} segments...")

            self.update_train_progress(95)
            self.train_log_message("=" * 50)
            self.train_log_message(f"PREPROCESSING COMPLETE!")
            self.train_log_message(f"  Segments saved: {saved_count}")
            self.train_log_message(f"  Output folder: {output_folder}")
            self.train_log_message("=" * 50)
            self.train_log_message("")
            self.train_log_message("Now switch to 'Folder with pairs' mode and select the output folder,")
            self.train_log_message("or the folder has been auto-selected for you.")

            # Auto-select the output folder and switch to folder mode
            self.train_folder_var.set(output_folder)
            self.data_source_mode.set("folder")
            self.root.after(0, self.on_data_source_change)
            self.root.after(100, self.scan_training_folder)

            self.update_train_progress(100)
            self.train_status_var.set(f"Preprocessing complete: {saved_count} segments")

        except Exception as e:
            self.train_log_message(f"ERROR: {str(e)}")
            import traceback
            self.train_log_message(traceback.format_exc())
            self.train_status_var.set(f"Error: {str(e)[:50]}")

        finally:
            self.preprocessing_running = False
            self.root.after(0, lambda: self.preprocess_btn.configure(state="normal"))

    def scan_training_folder(self):
        """Scan folder for audio+transcript pairs"""
        folder = self.train_folder_var.get()
        if not folder or not os.path.isdir(folder):
            self.folder_info_var.set("Invalid folder path")
            return

        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
        pairs = []

        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            name, ext = os.path.splitext(file)

            if ext.lower() in audio_extensions:
                txt_path = os.path.join(folder, name + '.txt')
                if os.path.exists(txt_path):
                    pairs.append((filepath, txt_path))

        self.training_pairs = pairs
        self.folder_info_var.set(f"Found {len(pairs)} audio+transcript pairs")

        # Update sample count max and auto-fill voice name
        if pairs:
            max_samples = len(pairs)
            self.sample_spin.configure(to=max_samples)
            # Auto-set to ALL samples (not just 10)
            self.sample_count_var.set(max_samples)

            # Auto-fill voice name from folder name if empty
            if not self.voice_name_var.get().strip():
                folder_name = os.path.basename(folder.rstrip('/\\'))
                # Clean up common suffixes
                for suffix in ['_segments', '_training', '_data']:
                    if folder_name.lower().endswith(suffix):
                        folder_name = folder_name[:-len(suffix)]
                self.voice_name_var.set(folder_name)

            self.train_log_message(f"Found {len(pairs)} training samples in {folder}")
            self.train_log_message(f"Voice name: {self.voice_name_var.get()}, Samples: {max_samples}")
        else:
            self.train_log_message("No valid audio+transcript pairs found. Make sure each audio file has a matching .txt file.")

    def generate_training_jsonl(self, output_path: str, max_samples: int) -> str:
        """Generate JSONL file for training from audio+transcript pairs"""
        records = []
        for audio_path, txt_path in self.training_pairs[:max_samples]:
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                records.append({
                    "audio": audio_path,
                    "text": text,
                    "language": "en"
                })
            except Exception as e:
                self.train_log_message(f"Warning: Could not read {txt_path}: {e}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return output_path

    def start_training(self):
        """Start the voice training process"""
        if self.training_running:
            return

        # Validate inputs
        voice_name = self.voice_name_var.get().strip()
        if not voice_name:
            messagebox.showerror("Error", "Please enter a voice name")
            return

        if not self.training_pairs:
            messagebox.showerror("Error", "Please scan a folder with training data first")
            return

        # Sanitize voice name for filesystem
        voice_name = re.sub(r'[^\w\-_]', '_', voice_name)

        # Update UI state
        self.training_running = True
        self.start_train_btn.configure(state="disabled")
        self.cancel_train_btn.configure(state="normal")
        self.train_progress.set(0)

        # Clear log
        self.train_log.configure(state="normal")
        self.train_log.delete("1.0", tk.END)
        self.train_log.configure(state="disabled")

        # Start training thread
        self.train_thread = threading.Thread(
            target=self.run_training_pipeline,
            args=(voice_name, self.sample_count_var.get(), self.epochs_var.get()),
            daemon=True
        )
        self.train_thread.start()

        # Start polling for training output
        self.poll_training_output()

    def run_training_pipeline(self, voice_name: str, sample_count: int, epochs: int):
        """Run the complete training pipeline in background thread"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            voices_dir = os.path.join(base_dir, TRAINED_VOICES_DIR)
            os.makedirs(voices_dir, exist_ok=True)

            output_dir = os.path.join(voices_dir, voice_name)
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Step 1: Generate JSONL
            self.train_log_message("Step 1/4: Generating training data file...")
            self.update_train_progress(5)
            self.train_status_var.set("Step 1/4: Preparing data...")

            input_jsonl = os.path.join(temp_dir, "train_input.jsonl")
            self.generate_training_jsonl(input_jsonl, sample_count)
            self.train_log_message(f"  Created {input_jsonl} with {sample_count} samples")

            if not self.training_running:
                self.train_log_message("Training cancelled.")
                return

            # Step 2: Run prepare_data.py
            self.train_log_message("Step 2/4: Encoding audio to tokens...")
            self.update_train_progress(15)
            self.train_status_var.set("Step 2/4: Encoding audio...")

            prepared_jsonl = os.path.join(temp_dir, "train_prepared.jsonl")
            prepare_script = os.path.join(base_dir, "MOSS-TTS", "moss_tts_local", "finetuning", "prepare_data.py")

            # Get python executable from venv
            python_exe = os.path.join(base_dir, "venv_voice_ai", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = sys.executable

            prepare_cmd = [
                python_exe, prepare_script,
                "--model-path", "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
                "--codec-path", "OpenMOSS-Team/MOSS-Audio-Tokenizer",
                "--input-jsonl", input_jsonl,
                "--output-jsonl", prepared_jsonl,
            ]

            self.run_subprocess_with_logging(prepare_cmd, "prepare_data")

            if not self.training_running:
                self.train_log_message("Training cancelled.")
                return

            # Step 3: Run sft.py
            self.train_log_message("Step 3/4: Fine-tuning model (this may take a while)...")
            self.update_train_progress(30)
            self.train_status_var.set("Step 3/4: Fine-tuning...")

            sft_script = os.path.join(base_dir, "MOSS-TTS", "moss_tts_local", "finetuning", "sft.py")

            sft_cmd = [
                python_exe, sft_script,
                "--model-path", "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
                "--codec-path", "OpenMOSS-Team/MOSS-Audio-Tokenizer",
                "--train-jsonl", prepared_jsonl,
                "--output-dir", output_dir,
                "--num-epochs", str(epochs),
                "--per-device-batch-size", "1",
                "--gradient-accumulation-steps", "4",
                "--learning-rate", "1e-5",
                "--mixed-precision", "bf16",
                "--gradient-checkpointing",
            ]

            self.run_subprocess_with_logging(sft_cmd, "sft", progress_range=(30, 90))

            if not self.training_running:
                self.train_log_message("Training cancelled.")
                return

            # Step 4: Clean up and finalize
            self.train_log_message("Step 4/4: Finalizing trained voice...")
            self.update_train_progress(95)
            self.train_status_var.set("Step 4/4: Finalizing...")

            # Create voice metadata
            metadata = {
                "name": voice_name,
                "samples_used": sample_count,
                "epochs": epochs,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_path": output_dir
            }

            with open(os.path.join(output_dir, "voice_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            self.update_train_progress(100)
            self.train_log_message(f"Training complete! Voice '{voice_name}' saved to {output_dir}")
            self.train_status_var.set(f"Complete: Voice '{voice_name}' ready")

            # Refresh voice list in Chat tab
            self.root.after(0, self.refresh_trained_voices)

        except Exception as e:
            self.train_log_message(f"ERROR: {str(e)}")
            import traceback
            self.train_log_message(traceback.format_exc())
            self.train_status_var.set(f"Error: {str(e)[:50]}")

        finally:
            self.training_running = False
            self.root.after(0, self.on_training_complete)

    def run_subprocess_with_logging(self, cmd, step_name, progress_range=None):
        """Run a subprocess and log its output in real-time"""
        self.train_log_message(f"  Running: {' '.join(cmd[:3])}...")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            for line in iter(process.stdout.readline, ''):
                if not self.training_running:
                    process.terminate()
                    return

                line = line.strip()
                if line:
                    self.train_log_message(f"  [{step_name}] {line}")

                    # Parse progress from sft.py output
                    if progress_range and ("step" in line.lower() or "epoch" in line.lower()):
                        try:
                            # Try to extract progress info
                            match = re.search(r'(\d+)/(\d+)', line)
                            if match:
                                current, total = int(match.group(1)), int(match.group(2))
                                if total > 0:
                                    start, end = progress_range
                                    progress = start + (end - start) * (current / total)
                                    self.update_train_progress(progress)
                        except:
                            pass

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"{step_name} failed with return code {process.returncode}")

        except FileNotFoundError:
            raise RuntimeError(f"Could not find script: {cmd[1]}")

    def train_log_message(self, message: str):
        """Add message to training log (thread-safe via queue)"""
        self.train_output_queue.put(("log", message))

    def update_train_progress(self, value: float):
        """Update progress bar (thread-safe via queue)"""
        self.train_output_queue.put(("progress", value))

    def poll_training_output(self):
        """Poll for training output updates"""
        try:
            while True:
                msg_type, data = self.train_output_queue.get_nowait()
                if msg_type == "log":
                    self.train_log.configure(state="normal")
                    self.train_log.insert(tk.END, data + "\n")
                    self.train_log.see(tk.END)
                    self.train_log.configure(state="disabled")
                elif msg_type == "progress":
                    self.train_progress.set(data / 100.0)
        except queue.Empty:
            pass

        if self.training_running or self.preprocessing_running:
            self.root.after(100, self.poll_training_output)

    def cancel_training(self):
        """Cancel the current training operation"""
        if self.training_running:
            self.training_running = False
            self.train_log_message("Training cancelled by user")
            self.train_status_var.set("Cancelled")

    def on_training_complete(self):
        """Called when training finishes or is cancelled"""
        self.start_train_btn.configure(state="normal")
        self.cancel_train_btn.configure(state="disabled")

    def refresh_trained_voices(self):
        """Refresh the voice dropdown with any newly trained voices"""
        self.trained_voice_paths = get_trained_voices()

        # Rebuild voice options
        all_voices = list(VOICE_OPTIONS.keys())

        # Insert trained voices before "Custom Audio File..."
        for voice_name in self.trained_voice_paths.keys():
            if voice_name not in all_voices:
                all_voices.insert(-1, voice_name)

        # Update combobox
        self.voice_combo.configure(values=all_voices)

    def on_closing(self):
        """Handle window close"""
        if self.chat_running:
            self.stop_chat()
            self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()


class SimplifiedVoiceChat:
    """Simplified voice chat that reports to the UI"""

    def __init__(self, output_queue, system_prompt, whisper_model, input_mode, ollama_model, voice_path, fast_mode, trained_model_path=None, streaming_mode=True, tts_engine="MOSS-TTS", qwen3_speaker="serena", omni_language="en", omni_voice_design="", omni_speed=1.0):
        import torch
        self.output_queue = output_queue
        self.system_prompt = system_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.input_mode = input_mode
        self.ollama_model = ollama_model
        self.voice_path = voice_path
        self.fast_mode = fast_mode
        self.trained_model_path = trained_model_path  # Path to fine-tuned model checkpoint
        self.streaming_mode = streaming_mode  # Enable streaming LLM->TTS for faster first response
        self.running = True
        self.conversation_history = []

        # TTS engine settings
        self.tts_engine_name = tts_engine
        self.qwen3_speaker = qwen3_speaker
        self.omni_language = omni_language
        self.omni_voice_design = omni_voice_design
        self.omni_speed = omni_speed
        self.tts_engine = None  # Will be loaded lazily

        # Models
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.tts_model = None
        self.tts_inferencer = None
        self.tts_codec = None
        self.audio_recorder = None

        # Load audio recorder with level callback
        from audio_utils import AudioRecorder
        self.audio_recorder = AudioRecorder()
        self.audio_recorder.level_callback = self.on_audio_level

        import config
        self.config = config

    def on_audio_level(self, level):
        """Callback for audio level updates"""
        self.output_queue.put(("level", str(level)))

    def log(self, tag, message):
        self.output_queue.put((tag, message))

    def preload_all_models(self):
        """Preload all models into VRAM at startup for zero-delay operation"""
        import requests

        # 1. Load WhisperX
        self.log("status", "Initializing: Loading WhisperX...")
        self.load_whisper()

        # 2. Warm up Ollama LLM (trigger model load)
        self.log("status", "Initializing: Loading LLM...")
        try:
            # Send a minimal request to force Ollama to load the model
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=120  # Model loading can take time
            )
            response.raise_for_status()
            self.log("system", f"LLM {self.ollama_model} loaded")
        except Exception as e:
            self.log("error", f"Warning: Could not warm up LLM: {e}")

        # 3. Load TTS Engine
        self.log("status", f"Initializing: Loading {self.tts_engine_name}...")
        self.load_tts()

        self.log("system", "All models loaded - Ready!")

    def load_whisper(self):
        if self.whisper_model is None:
            self.log("status", "Loading WhisperX...")
            import whisperx
            # Try float16 first, fall back to int8 or float32 if not supported
            for compute_type in ["float16", "int8", "float32"]:
                try:
                    self.whisper_model = whisperx.load_model(
                        self.whisper_model_name,
                        self.device,
                        compute_type=compute_type
                    )
                    self.log("system", f"WhisperX {self.whisper_model_name} loaded (compute_type={compute_type})")
                    break
                except ValueError as e:
                    if "compute type" in str(e).lower() and compute_type != "float32":
                        self.log("system", f"WhisperX: {compute_type} not supported, trying next...")
                        continue
                    raise

    def load_tts(self):
        if self.tts_engine is None:
            import torch
            self.log("status", f"Loading {self.tts_engine_name}...")

            from tts_engines import create_tts_engine

            # Create the TTS engine based on selection
            if self.tts_engine_name == "Qwen3-TTS":
                self.tts_engine = create_tts_engine(
                    self.tts_engine_name,
                    device=self.device,
                    dtype=self.dtype,
                    speaker=self.qwen3_speaker
                )
            elif self.tts_engine_name == "OmniVoice":
                # Get absolute voice path for pre-loading clone prompt
                omni_voice_path = None
                if self.voice_path and self.voice_path != "custom":
                    vp = os.path.abspath(self.voice_path) if os.path.exists(str(self.voice_path)) else None
                    if vp:
                        omni_voice_path = vp
                self.tts_engine = create_tts_engine(
                    self.tts_engine_name,
                    device=self.device,
                    dtype=self.dtype,
                    language=self.omni_language,
                    voice_design=self.omni_voice_design,
                    speed=self.omni_speed,
                    voice_path=omni_voice_path
                )
            else:
                # MOSS-TTS
                self.tts_engine = create_tts_engine(
                    self.tts_engine_name,
                    device=self.device,
                    dtype=self.dtype
                )

            # Load the engine
            self.tts_engine.load()

            # Log model info
            self.log("system", f"{self.tts_engine_name} loaded on {self.device}")
            if self.tts_engine_name == "Qwen3-TTS":
                self.log("system", f"Speaker: {self.qwen3_speaker}")
            elif self.tts_engine_name == "OmniVoice":
                self.log("system", f"Language: {self.omni_language}, Speed: {self.omni_speed}")
                if self.omni_voice_design:
                    self.log("system", f"Voice Design: {self.omni_voice_design}")

    def transcribe(self, audio):
        import numpy as np
        import soundfile as sf
        import tempfile
        import os
        import warnings

        self.load_whisper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000)
            temp_path = f.name

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.whisper_model.transcribe(
                    temp_path,
                    batch_size=16,
                    language="en"
                )

            if not result or "segments" not in result or not result["segments"]:
                self.log("system", "No speech detected in audio")
                return ""

            text = " ".join([seg["text"] for seg in result["segments"]])
            return text.strip()
        except IndexError:
            self.log("system", "No speech detected in audio")
            return ""
        except Exception as e:
            self.log("error", f"Transcription error: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def chat_ollama(self, user_message):
        import requests

        self.conversation_history.append({"role": "user", "content": user_message})

        # Build system prompt - add instruction for concise responses
        effective_system_prompt = self.system_prompt
        if self.fast_mode:
            effective_system_prompt += "\n\nRespond briefly in 1-2 sentences. Be direct and conversational."

        # Add OmniVoice non-verbal expression instructions when using OmniVoice TTS
        if self.tts_engine_name == "OmniVoice":
            effective_system_prompt += (
                "\n\nIMPORTANT: Your text will be spoken aloud using an expressive TTS engine. "
                "You can use these inline non-verbal tags naturally in your responses to make speech more expressive: "
                "[laughter], [sigh], [confirmation-en], [question-en], [question-ah], [question-oh], "
                "[surprise-ah], [surprise-oh], [surprise-wa], [dissatisfaction-hnn]. "
                "Use them sparingly and naturally, e.g. '[laughter] That's a great point.' or "
                "'[sigh] I know what you mean.' "
                "For English pronunciation corrections, use CMU dictionary format in brackets like [IH1 T] for 'it'."
            )

        messages = [{"role": "system", "content": effective_system_prompt}]
        messages.extend(self.conversation_history[-10:])

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "think": False,  # Disable thinking mode for faster responses
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            # Debug: show full response structure
            self.log("system", f"[DEBUG] Ollama response keys: {result.keys()}")
            self.log("system", f"[DEBUG] Message: {result.get('message', {})}")

            assistant_message = result.get("message", {}).get("content", "")

            # Debug: show raw response
            self.log("system", f"[DEBUG] Raw LLM response ({len(assistant_message)} chars): '{assistant_message[:200]}'")

            # Clean up any thinking tags that might still appear
            if "<think>" in assistant_message:
                # Remove thinking section
                import re
                assistant_message = re.sub(r'<think>.*?</think>', '', assistant_message, flags=re.DOTALL)
                assistant_message = assistant_message.strip()
                self.log("system", f"[DEBUG] After cleaning: ({len(assistant_message)} chars)")

            # If still empty, provide a fallback
            if not assistant_message:
                assistant_message = "I'm sorry, I didn't generate a proper response. Could you try again?"

            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            return f"Sorry, I couldn't connect to the language model: {e}"

    def synthesize_speech(self, text):
        import numpy as np

        self.load_tts()

        # Get absolute path for voice reference
        voice_ref = None
        if self.voice_path and self.voice_path != "custom":
            voice_ref = os.path.abspath(self.voice_path) if os.path.exists(self.voice_path) else None

        # Detailed TTS timing
        t_gen_start = time.time()

        self.log("system", f"[DEBUG] Starting TTS ({self.tts_engine_name}) for: '{text[:50]}...'")

        # Use the TTS engine abstraction
        audio = self.tts_engine.synthesize(text, voice_path=voice_ref)

        t_gen_end = time.time()
        gen_time = t_gen_end - t_gen_start

        self.log("system", f"[DEBUG] TTS completed in {gen_time:.2f}s")

        return audio if len(audio) > 0 else np.array([])

    def stream_chat_ollama(self, user_message):
        """Stream LLM response and yield text chunks at sentence boundaries"""
        import re
        import json

        self.conversation_history.append({"role": "user", "content": user_message})

        # Build effective system prompt with TTS-specific instructions
        effective_prompt = self.system_prompt
        if self.tts_engine_name == "OmniVoice":
            effective_prompt += (
                "\n\nIMPORTANT: Your text will be spoken aloud using an expressive TTS engine. "
                "You can use these inline non-verbal tags naturally in your responses to make speech more expressive: "
                "[laughter], [sigh], [confirmation-en], [question-en], [question-ah], [question-oh], "
                "[surprise-ah], [surprise-oh], [surprise-wa], [dissatisfaction-hnn]. "
                "Use them sparingly and naturally, e.g. '[laughter] That's a great point.' or "
                "'[sigh] I know what you mean.' "
                "For English pronunciation corrections, use CMU dictionary format in brackets like [IH1 T] for 'it'."
            )

        messages = [{"role": "system", "content": effective_prompt}]
        messages.extend(self.conversation_history[-10:])

        self.log("system", "[DEBUG] Starting LLM stream...")

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "options": {"temperature": 0.7, "num_predict": 512}
                },
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            buffer = ""
            full_response = ""
            # Split on sentence boundaries: . ! ? followed by space, or paragraph breaks
            split_pattern = re.compile(r'([.!?])\s+|\n\n+')
            min_chunk_len = 50  # Minimum characters for a chunk (avoid tiny chunks)
            chunk_num = 0

            for line in response.iter_lines():
                if not self.running:
                    break
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            buffer += token
                            full_response += token

                            while True:
                                match = split_pattern.search(buffer)
                                if match:
                                    end_pos = match.end()
                                    chunk = buffer[:end_pos].strip()
                                    # Require minimum length to avoid tiny chunks
                                    if len(chunk) < min_chunk_len:
                                        break
                                    if chunk:
                                        chunk_num += 1
                                        self.log("system", f"[DEBUG] LLM chunk {chunk_num}: '{chunk[:50]}...'")
                                        yield chunk
                                    buffer = buffer[end_pos:]
                                else:
                                    break

                            if data.get("done", False):
                                break
                    except json.JSONDecodeError:
                        continue

            # Yield remaining buffer - split into sentence-sized chunks
            while buffer.strip():
                match = split_pattern.search(buffer)
                if match and len(buffer[:match.end()].strip()) >= min_chunk_len:
                    end_pos = match.end()
                    chunk = buffer[:end_pos].strip()
                    if chunk:
                        chunk_num += 1
                        self.log("system", f"[DEBUG] LLM chunk {chunk_num}: '{chunk[:50]}...'")
                        yield chunk
                    buffer = buffer[end_pos:]
                else:
                    # No more valid split points, yield what's left
                    if buffer.strip():
                        chunk_num += 1
                        self.log("system", f"[DEBUG] LLM final chunk {chunk_num}: '{buffer.strip()[:50]}...'")
                        yield buffer.strip()
                    break

            self.log("system", f"[DEBUG] LLM stream complete. Total chunks: {chunk_num}")
            self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            self.log("error", f"Ollama streaming error: {e}")
            yield "Sorry, I couldn't connect to the language model."

    def process_turn_streaming(self):
        """Process one turn with streaming LLM->TTS for faster first response"""
        import numpy as np
        import queue
        import threading
        import sounddevice as sd
        import torch
        import keyboard

        self.log("system", "[DEBUG] Using STREAMING mode (parallel)")

        if not self.running:
            return

        # Record
        self.log("status", "Running - Hold SPACE to speak" if self.input_mode == "ptt" else "Running - Listening...")

        if self.input_mode == "ptt":
            audio = self.audio_recorder.record_ptt()
        else:
            audio = self.audio_recorder.record_vad()

        self.output_queue.put(("level", "0"))

        if not self.running:
            return

        if len(audio) < 16000 * 0.5:
            self.log("system", "Recording too short, try again")
            return

        t_start = time.time()

        # Transcribe
        self.log("status", "Transcribing...")
        t_stt_start = time.time()
        user_text = self.transcribe(audio)
        stt_time = time.time() - t_stt_start

        if not self.running or not user_text:
            self.log("status", "Running - Hold SPACE to speak" if self.input_mode == "ptt" else "Running - Listening...")
            return

        self.log("user", user_text)
        self.log("system", f"[DEBUG] STT: {stt_time:.2f}s")

        # Streaming LLM->TTS with parallel processing
        self.log("status", "Responding (streaming)... [SPACE to interrupt]")
        self.load_tts()

        # Queue for text chunks from LLM -> TTS processor
        text_queue = queue.Queue()
        # Queue for audio chunks from TTS -> audio player
        audio_queue = queue.Queue()
        playback_done = threading.Event()
        interrupted = threading.Event()  # Interrupt signal
        full_response_parts = []
        t_first_audio = [None]  # Use list to allow mutation in thread
        turn_seed = int(time.time()) % 10000

        def tts_processor():
            """Process text chunks into audio in a separate thread"""
            chunk_count = 0
            while not interrupted.is_set():
                try:
                    item = text_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:  # Sentinel to stop
                    break
                if interrupted.is_set():
                    break
                text_chunk = item
                chunk_count += 1

                try:
                    torch.manual_seed(turn_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(turn_seed)

                    t_tts_start = time.time()
                    self.log("system", f"[DEBUG] TTS chunk {chunk_count} starting: '{text_chunk[:30]}...'")
                    audio_chunk = self.synthesize_speech(text_chunk)
                    t_tts_end = time.time()

                    if interrupted.is_set():
                        break

                    if len(audio_chunk) > 0:
                        if t_first_audio[0] is None:
                            t_first_audio[0] = time.time()
                            latency = t_first_audio[0] - t_start
                            self.log("system", f"[DEBUG] First audio ready: {latency:.2f}s (TTS: {t_tts_end-t_tts_start:.2f}s)")
                        else:
                            self.log("system", f"[DEBUG] TTS chunk {chunk_count} done: {t_tts_end-t_tts_start:.2f}s")
                        audio_queue.put(audio_chunk)
                except Exception as e:
                    self.log("error", f"TTS error: {e}")

            audio_queue.put(None)  # Signal audio player to stop

        def audio_player():
            """Play audio chunks as they become available"""
            sample_rate = self.tts_engine.get_sample_rate() if self.tts_engine else 24000
            self.log("system", f"[DEBUG] Audio player started, sample_rate={sample_rate}")
            while True:
                if interrupted.is_set():
                    sd.stop()
                    self.log("system", "[DEBUG] Audio player interrupted")
                    break
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if audio_chunk is None:
                    self.log("system", "[DEBUG] Audio player received stop signal")
                    break
                if interrupted.is_set():
                    sd.stop()
                    break
                try:
                    self.log("system", f"[DEBUG] Playing audio: {len(audio_chunk)} samples, {len(audio_chunk)/sample_rate:.2f}s")
                    sd.play(audio_chunk, sample_rate)
                    # Poll for interrupt during playback instead of blocking
                    while sd.get_stream().active:
                        if interrupted.is_set():
                            sd.stop()
                            self.log("system", "[DEBUG] Playback interrupted mid-audio")
                            break
                        time.sleep(0.05)
                except Exception as e:
                    self.log("error", f"Playback error: {e}")
            playback_done.set()

        # Start TTS processor and audio player threads
        tts_thread = threading.Thread(target=tts_processor, daemon=True)
        player_thread = threading.Thread(target=audio_player, daemon=True)
        tts_thread.start()
        player_thread.start()

        # Stream LLM and feed text chunks to TTS processor
        chunk_count = 0
        for text_chunk in self.stream_chat_ollama(user_text):
            if not self.running or interrupted.is_set():
                break
            full_response_parts.append(text_chunk)
            chunk_count += 1
            self.log("system", f"[DEBUG] LLM->TTS queue chunk {chunk_count}")
            text_queue.put(text_chunk)

        # Signal TTS processor to finish
        text_queue.put(None)

        # Log full response
        full_response = " ".join(full_response_parts)
        self.log("assistant", full_response)

        # Monitor for interrupt while waiting for playback
        while not playback_done.is_set():
            if keyboard.is_pressed("space") and not interrupted.is_set():
                interrupted.set()
                sd.stop()
                self.log("system", "[Interrupted by user]")
                # Drain queues
                while not text_queue.empty():
                    try:
                        text_queue.get_nowait()
                    except queue.Empty:
                        break
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
                audio_queue.put(None)  # Ensure player exits
                break
            if not self.running:
                interrupted.set()
                sd.stop()
                break
            time.sleep(0.05)

        playback_done.wait(timeout=5)

        total_time = time.time() - t_start
        if interrupted.is_set():
            self.log("system", f"[DEBUG] TOTAL (interrupted): {total_time:.2f}s ({chunk_count} chunks)")
        else:
            self.log("system", f"[DEBUG] TOTAL: {total_time:.2f}s ({chunk_count} chunks)")
        self.log("status", "Running - Hold SPACE to speak" if self.input_mode == "ptt" else "Running - Listening...")

    def process_turn(self):
        import numpy as np
        from audio_utils import play_audio

        if not self.running:
            return

        # Record
        self.log("status", "Running - Hold SPACE to speak" if self.input_mode == "ptt" else "Running - Listening...")

        if self.input_mode == "ptt":
            audio = self.audio_recorder.record_ptt()
        else:
            audio = self.audio_recorder.record_vad()

        # Reset level
        self.output_queue.put(("level", "0"))

        if not self.running:
            return

        if len(audio) < 16000 * 0.5:
            self.log("system", "Recording too short, try again")
            return

        # ========== TIMING: Start ==========
        t_start = time.time()

        # Transcribe
        self.log("status", "Transcribing...")
        t_stt_start = time.time()
        user_text = self.transcribe(audio)
        t_stt_end = time.time()
        stt_time = t_stt_end - t_stt_start

        if not self.running:
            return

        if not user_text:
            self.log("status", "Running - Hold SPACE to speak" if self.input_mode == "ptt" else "Running - Listening...")
            return

        self.log("user", user_text)
        self.log("system", f"[DEBUG] STT: {stt_time:.2f}s")

        # Get response
        self.log("status", "Thinking...")
        t_llm_start = time.time()
        response_text = self.chat_ollama(user_text)
        t_llm_end = time.time()
        llm_time = t_llm_end - t_llm_start

        self.log("assistant", response_text)
        self.log("system", f"[DEBUG] LLM: {llm_time:.2f}s ({len(response_text)} chars)")

        if not self.running:
            return

        # Speak
        self.log("status", "Speaking... [SPACE to interrupt]")
        t_tts_start = time.time()
        tts_time = 0
        play_time = 0
        was_interrupted = False
        try:
            import sounddevice as sd
            import keyboard
            audio_response = self.synthesize_speech(response_text)
            t_tts_end = time.time()
            tts_time = t_tts_end - t_tts_start
            self.log("system", f"[DEBUG] TTS: {tts_time:.2f}s")

            if len(audio_response) > 0:
                t_play_start = time.time()
                sample_rate = self.tts_engine.get_sample_rate() if self.tts_engine else 24000
                sd.play(audio_response, sample_rate)
                # Poll for interrupt during playback
                while sd.get_stream().active:
                    if keyboard.is_pressed("space"):
                        sd.stop()
                        was_interrupted = True
                        self.log("system", "[Interrupted by user]")
                        break
                    if not self.running:
                        sd.stop()
                        break
                    time.sleep(0.05)
                t_play_end = time.time()
                play_time = t_play_end - t_play_start
                self.log("system", f"[DEBUG] Playback: {play_time:.2f}s{' (interrupted)' if was_interrupted else ''}")
        except Exception as e:
            self.log("error", f"TTS error: {e}")
            tts_time = time.time() - t_tts_start

        # Total time summary
        t_end = time.time()
        total_time = t_end - t_start
        self.log("system", f"[DEBUG] TOTAL: {total_time:.2f}s (STT:{stt_time:.2f} + LLM:{llm_time:.2f} + TTS:{tts_time:.2f})")

        self.log("status", "Running - Hold SPACE to speak" if self.input_mode == "ptt" else "Running - Listening...")

    def cleanup(self):
        """Clean up all resources and free GPU memory"""
        import gc
        import torch

        self.log("system", "[DEBUG] Cleaning up resources...")

        # Stop audio recorder
        if self.audio_recorder:
            self.audio_recorder.stop_stream()
            self.audio_recorder = None

        # Unload TTS engine
        if self.tts_engine:
            self.log("system", "[DEBUG] Unloading TTS engine...")
            self.tts_engine.unload()
            self.tts_engine = None

        # Unload WhisperX model
        if self.whisper_model:
            self.log("system", "[DEBUG] Unloading WhisperX model...")
            del self.whisper_model
            self.whisper_model = None

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        self.log("system", "[DEBUG] Cleanup complete, GPU memory freed")


def main():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = VoiceChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
