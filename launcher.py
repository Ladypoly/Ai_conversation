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

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import sys
import json
import time
import requests
import re
import subprocess

CONFIG_FILE = "user_config.json"

# TTS Engine options
TTS_ENGINE_OPTIONS = ["MOSS-TTS", "Qwen3-TTS"]
QWEN3_SPEAKERS = ["serena", "aiden", "dylan", "eric", "ono_anna", "ryan", "sohee", "uncle_fu", "vivian"]

DEFAULT_PERSONAS = {
    "Helpful Assistant": "You are a helpful voice assistant. Keep your responses concise and conversational - aim for 1-3 sentences unless more detail is needed. Be friendly and natural.",
    "Coding Buddy": "You are a friendly coding assistant. Help with programming questions, explain concepts simply, and suggest solutions. Keep responses brief and conversational since this is a voice interface.",
    "Language Tutor": "You are a patient language tutor. Help practice conversation, correct mistakes gently, and explain grammar when asked. Speak naturally and encourage the learner.",
    "Storyteller": "You are a creative storyteller. Tell engaging short stories, continue narratives, and bring characters to life with expressive speech. Keep each response to a few sentences to maintain flow.",
    "Trivia Host": "You are an enthusiastic trivia game host. Ask interesting questions, give hints when needed, and celebrate correct answers. Keep the energy fun and engaging.",
    "Custom": ""
}

# Voice options - auto-discovered from MOSS-TTS audio folder
VOICE_AUDIO_DIR = "MOSS-TTS/moss_tts_realtime/audio"

def get_voice_options():
    """Scan audio folder and build voice options dict"""
    options = {}

    # Scan for audio files
    if os.path.exists(VOICE_AUDIO_DIR):
        audio_files = []
        for f in os.listdir(VOICE_AUDIO_DIR):
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(f)

        # Sort and add to options
        for i, f in enumerate(sorted(audio_files)):
            name = os.path.splitext(f)[0]  # Remove extension
            # Make friendly name
            display_name = name.replace('_', ' ').title()
            options[display_name] = os.path.join(VOICE_AUDIO_DIR, f)

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
        self.train_output_queue = queue.Queue()
        self.training_pairs = []
        self.trained_voice_paths = {}

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
            "qwen3_speaker": self.qwen3_speaker_var.get()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Chat
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="Chat")

        # Tab 2: Train Voice
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Train Voice")

        # Build widgets for each tab
        self._create_chat_widgets()
        self._create_train_widgets()

    def _create_chat_widgets(self):
        """Create widgets for the Chat tab"""
        # Top frame - Settings
        self.settings_frame = ttk.LabelFrame(self.chat_frame, text="Settings", padding="5")
        self.settings_frame.pack(fill=tk.X, pady=2)

        # Row 1 - Persona and Models
        row1 = ttk.Frame(self.settings_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Persona:").pack(side=tk.LEFT)
        self.persona_var = tk.StringVar(value=self.saved_config.get("persona", "Helpful Assistant"))
        persona_combo = ttk.Combobox(row1, textvariable=self.persona_var,
                                      values=list(DEFAULT_PERSONAS.keys()), state="readonly", width=18)
        persona_combo.pack(side=tk.LEFT, padx=5)
        persona_combo.bind("<<ComboboxSelected>>", self.on_persona_change)

        ttk.Label(row1, text="LLM:").pack(side=tk.LEFT, padx=(10, 0))
        self.ollama_var = tk.StringVar(value=self.saved_config.get("ollama_model", "qwen3.5:4b"))
        self.ollama_combo = ttk.Combobox(row1, textvariable=self.ollama_var,
                                          values=get_ollama_models(), width=18)
        self.ollama_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Whisper:").pack(side=tk.LEFT, padx=(10, 0))
        self.whisper_var = tk.StringVar(value=self.saved_config.get("whisper_model", "large-v3"))
        whisper_combo = ttk.Combobox(row1, textvariable=self.whisper_var,
                                      values=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                                      state="readonly", width=10)
        whisper_combo.pack(side=tk.LEFT, padx=5)

        # Row 2 - Voice and Mode
        row2 = ttk.Frame(self.settings_frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="Voice:").pack(side=tk.LEFT)
        # Default to first available voice (or first key in VOICE_OPTIONS)
        default_voice = list(VOICE_OPTIONS.keys())[0] if VOICE_OPTIONS else "Random (No Clone)"
        self.voice_var = tk.StringVar(value=self.saved_config.get("voice", default_voice))
        self.voice_combo = ttk.Combobox(row2, textvariable=self.voice_var,
                                    values=list(VOICE_OPTIONS.keys()), state="readonly", width=20)
        self.voice_combo.pack(side=tk.LEFT, padx=5)
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_change)

        ttk.Label(row2, text="Mode:").pack(side=tk.LEFT, padx=(10, 0))
        self.mode_var = tk.StringVar(value=self.saved_config.get("input_mode", "ptt"))
        ttk.Radiobutton(row2, text="Push-to-Talk", variable=self.mode_var,
                        value="ptt").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row2, text="VAD", variable=self.mode_var,
                        value="vad").pack(side=tk.LEFT)

        # Fast mode checkbox
        self.fast_mode_var = tk.BooleanVar(value=self.saved_config.get("fast_mode", True))
        ttk.Checkbutton(row2, text="Fast Mode", variable=self.fast_mode_var).pack(side=tk.LEFT, padx=(20, 0))

        # Streaming mode checkbox (streams LLM->TTS for faster first response)
        self.streaming_mode_var = tk.BooleanVar(value=self.saved_config.get("streaming_mode", True))
        ttk.Checkbutton(row2, text="Streaming", variable=self.streaming_mode_var).pack(side=tk.LEFT, padx=(10, 0))

        # Row 3 - TTS Engine selection
        row3 = ttk.Frame(self.settings_frame)
        row3.pack(fill=tk.X, pady=2)

        ttk.Label(row3, text="TTS Engine:").pack(side=tk.LEFT)
        self.tts_engine_var = tk.StringVar(value=self.saved_config.get("tts_engine", "MOSS-TTS"))
        self.tts_engine_combo = ttk.Combobox(row3, textvariable=self.tts_engine_var,
                                              values=TTS_ENGINE_OPTIONS, state="readonly", width=12)
        self.tts_engine_combo.pack(side=tk.LEFT, padx=5)
        self.tts_engine_combo.bind("<<ComboboxSelected>>", self.on_tts_engine_change)

        # Qwen3-TTS Speaker selector (only visible when Qwen3-TTS selected)
        self.qwen3_speaker_label = ttk.Label(row3, text="Speaker:")
        self.qwen3_speaker_var = tk.StringVar(value=self.saved_config.get("qwen3_speaker", "serena"))
        self.qwen3_speaker_combo = ttk.Combobox(row3, textvariable=self.qwen3_speaker_var,
                                                 values=QWEN3_SPEAKERS, state="readonly", width=10)

        # Show/hide speaker based on current engine selection
        self.on_tts_engine_change()

        # System prompt
        prompt_frame = ttk.LabelFrame(self.settings_frame, text="System Prompt", padding="2")
        prompt_frame.pack(fill=tk.X, pady=2)

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=3,
                                                      font=('Consolas', 9))
        self.prompt_text.pack(fill=tk.X)

        saved_prompt = self.saved_config.get("system_prompt", "")
        if saved_prompt:
            self.prompt_text.insert("1.0", saved_prompt)
        else:
            self.prompt_text.insert("1.0", DEFAULT_PERSONAS[self.persona_var.get()])

        # Voice Level Indicator
        level_frame = ttk.Frame(self.chat_frame)
        level_frame.pack(fill=tk.X, pady=2)

        ttk.Label(level_frame, text="Mic Level:").pack(side=tk.LEFT)
        self.level_canvas = tk.Canvas(level_frame, width=300, height=20, bg="#333333", highlightthickness=1)
        self.level_canvas.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.level_bar = self.level_canvas.create_rectangle(0, 0, 0, 20, fill="#4CAF50", outline="")

        # Conversation frame
        conv_frame = ttk.LabelFrame(self.chat_frame, text="Conversation", padding="5")
        conv_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.conv_text = scrolledtext.ScrolledText(conv_frame, wrap=tk.WORD, state=tk.DISABLED,
                                                    font=('Consolas', 10))
        self.conv_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for colored text
        self.conv_text.tag_configure("user", foreground="#2196F3", font=('Consolas', 10, 'bold'))
        self.conv_text.tag_configure("assistant", foreground="#4CAF50", font=('Consolas', 10, 'bold'))
        self.conv_text.tag_configure("system", foreground="#9E9E9E", font=('Consolas', 9, 'italic'))
        self.conv_text.tag_configure("error", foreground="#F44336")

        # Bottom frame - Status and controls
        bottom_frame = ttk.Frame(self.chat_frame)
        bottom_frame.pack(fill=tk.X, pady=5)

        # Status indicator
        status_frame = ttk.Frame(bottom_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_indicator = tk.Canvas(status_frame, width=16, height=16)
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        self.status_circle = self.status_indicator.create_oval(2, 2, 14, 14, fill="gray")

        self.status_var = tk.StringVar(value="Ready - Configure settings and click Start")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=('Helvetica', 9))
        status_label.pack(side=tk.LEFT)

        # Buttons
        self.stop_btn = ttk.Button(bottom_frame, text="Stop", command=self.stop_chat, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=5)

        self.start_btn = ttk.Button(bottom_frame, text="Start", command=self.start_chat)
        self.start_btn.pack(side=tk.RIGHT, padx=5)

        self.clear_btn = ttk.Button(bottom_frame, text="Clear Log", command=self.clear_log)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)

        refresh_btn = ttk.Button(bottom_frame, text="Refresh LLMs", command=self.refresh_llms)
        refresh_btn.pack(side=tk.RIGHT, padx=5)

        # Load trained voices into dropdown
        self.refresh_trained_voices()

    def _create_train_widgets(self):
        """Create widgets for the Train Voice tab"""
        # Container frame
        train_container = ttk.Frame(self.train_frame, padding="10")
        train_container.pack(fill=tk.BOTH, expand=True)

        # --- Data Source Section ---
        data_frame = ttk.LabelFrame(train_container, text="Training Data", padding="10")
        data_frame.pack(fill=tk.X, pady=5)

        # Folder selector row
        folder_row = ttk.Frame(data_frame)
        folder_row.pack(fill=tk.X, pady=2)

        ttk.Label(folder_row, text="Audio Folder:").pack(side=tk.LEFT)
        self.train_folder_var = tk.StringVar()
        self.train_folder_entry = ttk.Entry(folder_row, textvariable=self.train_folder_var, width=50)
        self.train_folder_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(folder_row, text="Browse...", command=self.browse_train_folder).pack(side=tk.LEFT)

        # Folder info label
        self.folder_info_var = tk.StringVar(value="Select a folder containing audio files with matching .txt transcripts")
        ttk.Label(data_frame, textvariable=self.folder_info_var, font=('Helvetica', 9, 'italic')).pack(anchor=tk.W, pady=2)

        # File format info
        format_info = ttk.Label(data_frame, text="Expected format: audio1.wav + audio1.txt, audio2.mp3 + audio2.txt, etc.",
                                 font=('Helvetica', 8), foreground="gray")
        format_info.pack(anchor=tk.W)

        # --- Training Settings Section ---
        settings_frame = ttk.LabelFrame(train_container, text="Training Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)

        # Voice name
        name_row = ttk.Frame(settings_frame)
        name_row.pack(fill=tk.X, pady=2)
        ttk.Label(name_row, text="Voice Name:").pack(side=tk.LEFT)
        self.voice_name_var = tk.StringVar()
        ttk.Entry(name_row, textvariable=self.voice_name_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Label(name_row, text="(used to identify trained voice)", font=('Helvetica', 8), foreground="gray").pack(side=tk.LEFT)

        # Sample count selector
        samples_row = ttk.Frame(settings_frame)
        samples_row.pack(fill=tk.X, pady=2)
        ttk.Label(samples_row, text="Samples to use:").pack(side=tk.LEFT)
        self.sample_count_var = tk.IntVar(value=10)
        self.sample_spin = ttk.Spinbox(samples_row, from_=1, to=1000, textvariable=self.sample_count_var, width=10)
        self.sample_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(samples_row, text="(more samples = better quality, longer training)",
                  font=('Helvetica', 9, 'italic')).pack(side=tk.LEFT)

        # Epochs selector
        epochs_row = ttk.Frame(settings_frame)
        epochs_row.pack(fill=tk.X, pady=2)
        ttk.Label(epochs_row, text="Training Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=3)
        ttk.Spinbox(epochs_row, from_=1, to=10, textvariable=self.epochs_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(epochs_row, text="(more epochs = better fit, risk of overfitting)",
                  font=('Helvetica', 9, 'italic')).pack(side=tk.LEFT)

        # Model info (fixed)
        model_row = ttk.Frame(settings_frame)
        model_row.pack(fill=tk.X, pady=2)
        ttk.Label(model_row, text="Model: MossTTSLocal 1.7B", font=('Helvetica', 9)).pack(side=tk.LEFT)
        ttk.Label(model_row, text="(optimized for RTX 4090)", font=('Helvetica', 8), foreground="gray").pack(side=tk.LEFT, padx=5)

        # --- Progress Section ---
        progress_frame = ttk.LabelFrame(train_container, text="Training Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Progress bar
        self.train_progress_var = tk.DoubleVar(value=0)
        self.train_progress = ttk.Progressbar(progress_frame, variable=self.train_progress_var,
                                               maximum=100, mode='determinate')
        self.train_progress.pack(fill=tk.X, pady=5)

        # Status label
        self.train_status_var = tk.StringVar(value="Ready to train")
        ttk.Label(progress_frame, textvariable=self.train_status_var, font=('Helvetica', 9, 'bold')).pack(anchor=tk.W)

        # Log output
        self.train_log = scrolledtext.ScrolledText(progress_frame, wrap=tk.WORD, height=15,
                                                    font=('Consolas', 9), state=tk.DISABLED)
        self.train_log.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- Control Buttons ---
        btn_frame = ttk.Frame(train_container)
        btn_frame.pack(fill=tk.X, pady=10)

        self.scan_folder_btn = ttk.Button(btn_frame, text="Scan Folder", command=self.scan_training_folder)
        self.scan_folder_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_train_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel_training, state=tk.DISABLED)
        self.cancel_train_btn.pack(side=tk.RIGHT, padx=5)

        self.start_train_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training)
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
        """Handle TTS engine selection change - show/hide speaker selector"""
        engine = self.tts_engine_var.get()
        if engine == "Qwen3-TTS":
            # Show speaker selector
            self.qwen3_speaker_label.pack(side=tk.LEFT, padx=(15, 0))
            self.qwen3_speaker_combo.pack(side=tk.LEFT, padx=5)
        else:
            # Hide speaker selector
            self.qwen3_speaker_label.pack_forget()
            self.qwen3_speaker_combo.pack_forget()

    def refresh_llms(self):
        """Refresh the list of available Ollama models"""
        models = get_ollama_models()
        self.ollama_combo['values'] = models
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
        self.conv_text.configure(state=tk.NORMAL)
        if tag:
            self.conv_text.insert(tk.END, text, tag)
        else:
            self.conv_text.insert(tk.END, text)
        self.conv_text.see(tk.END)
        self.conv_text.configure(state=tk.DISABLED)

    def clear_log(self):
        """Clear conversation log"""
        self.conv_text.configure(state=tk.NORMAL)
        self.conv_text.delete("1.0", tk.END)
        self.conv_text.configure(state=tk.DISABLED)

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
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.prompt_text.configure(state=tk.DISABLED)
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

        # Start chat thread
        self.chat_running = True
        self.chat_thread = threading.Thread(
            target=self.run_chat_loop,
            args=(system_prompt, input_mode, whisper_model, ollama_model, voice_path, fast_mode, trained_model_path, tts_engine, qwen3_speaker),
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
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.prompt_text.configure(state=tk.NORMAL)

        self.output_queue.put(("system", "Chat stopped by user"))
        self.set_status("Stopped - Click Start to begin again", "gray")

    def run_chat_loop(self, system_prompt, input_mode, whisper_model, ollama_model, voice_path, fast_mode, trained_model_path=None, tts_engine="MOSS-TTS", qwen3_speaker="serena"):
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
                qwen3_speaker=qwen3_speaker
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
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.prompt_text.configure(state=tk.NORMAL)
        self.set_status("Stopped - Click Start to begin again", "gray")
        self.update_level(0)

    # ============== Training Methods ==============

    def browse_train_folder(self):
        """Open folder picker for training data"""
        folder = filedialog.askdirectory(title="Select folder with audio files and transcripts")
        if folder:
            self.train_folder_var.set(folder)
            self.scan_training_folder()

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

        # Update sample count max
        if pairs:
            max_samples = len(pairs)
            self.sample_spin.configure(to=max_samples)
            self.sample_count_var.set(min(max_samples, 10))
            self.train_log_message(f"Found {len(pairs)} training samples in {folder}")
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
        self.start_train_btn.configure(state=tk.DISABLED)
        self.cancel_train_btn.configure(state=tk.NORMAL)
        self.train_progress_var.set(0)

        # Clear log
        self.train_log.configure(state=tk.NORMAL)
        self.train_log.delete("1.0", tk.END)
        self.train_log.configure(state=tk.DISABLED)

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
                    self.train_log.configure(state=tk.NORMAL)
                    self.train_log.insert(tk.END, data + "\n")
                    self.train_log.see(tk.END)
                    self.train_log.configure(state=tk.DISABLED)
                elif msg_type == "progress":
                    self.train_progress_var.set(data)
        except queue.Empty:
            pass

        if self.training_running:
            self.root.after(100, self.poll_training_output)

    def cancel_training(self):
        """Cancel the current training operation"""
        if self.training_running:
            self.training_running = False
            self.train_log_message("Training cancelled by user")
            self.train_status_var.set("Cancelled")

    def on_training_complete(self):
        """Called when training finishes or is cancelled"""
        self.start_train_btn.configure(state=tk.NORMAL)
        self.cancel_train_btn.configure(state=tk.DISABLED)

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
        self.voice_combo['values'] = all_voices

    def on_closing(self):
        """Handle window close"""
        if self.chat_running:
            self.stop_chat()
            self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()


class SimplifiedVoiceChat:
    """Simplified voice chat that reports to the UI"""

    def __init__(self, output_queue, system_prompt, whisper_model, input_mode, ollama_model, voice_path, fast_mode, trained_model_path=None, streaming_mode=True, tts_engine="MOSS-TTS", qwen3_speaker="serena"):
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
            self.whisper_model = whisperx.load_model(
                self.whisper_model_name,
                self.device,
                compute_type="float16"
            )
            self.log("system", f"WhisperX {self.whisper_model_name} loaded")

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

        messages = [{"role": "system", "content": self.system_prompt}]
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
        self.log("status", "Responding (streaming)...")
        self.load_tts()

        # Queue for text chunks from LLM -> TTS processor
        text_queue = queue.Queue()
        # Queue for audio chunks from TTS -> audio player
        audio_queue = queue.Queue()
        playback_done = threading.Event()
        full_response_parts = []
        t_first_audio = [None]  # Use list to allow mutation in thread
        turn_seed = int(time.time()) % 10000

        def tts_processor():
            """Process text chunks into audio in a separate thread"""
            chunk_count = 0
            while True:
                item = text_queue.get()
                if item is None:  # Sentinel to stop
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
                audio_chunk = audio_queue.get()
                if audio_chunk is None:
                    self.log("system", "[DEBUG] Audio player received stop signal")
                    break
                try:
                    self.log("system", f"[DEBUG] Playing audio: {len(audio_chunk)} samples, {len(audio_chunk)/sample_rate:.2f}s")
                    sd.play(audio_chunk, sample_rate)
                    sd.wait()
                    self.log("system", "[DEBUG] Audio playback complete")
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
            if not self.running:
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

        # Wait for playback to complete
        playback_done.wait(timeout=120)

        total_time = time.time() - t_start
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
        self.log("status", "Speaking...")
        t_tts_start = time.time()
        tts_time = 0
        play_time = 0
        try:
            audio_response = self.synthesize_speech(response_text)
            t_tts_end = time.time()
            tts_time = t_tts_end - t_tts_start
            self.log("system", f"[DEBUG] TTS: {tts_time:.2f}s")

            if len(audio_response) > 0:
                t_play_start = time.time()
                sample_rate = self.tts_engine.get_sample_rate() if self.tts_engine else 24000
                play_audio(audio_response, sample_rate=sample_rate)
                t_play_end = time.time()
                play_time = t_play_end - t_play_start
                self.log("system", f"[DEBUG] Playback: {play_time:.2f}s")
        except Exception as e:
            self.log("error", f"TTS error: {e}")
            tts_time = time.time() - t_tts_start

        # Total time summary
        t_end = time.time()
        total_time = t_end - t_start
        self.log("system", f"[DEBUG] TOTAL: {total_time:.2f}s (STT:{stt_time:.2f} + LLM:{llm_time:.2f} + TTS:{tts_time:.2f})")

        self.log("status", "Running - Hold SPACE to speak" if self.input_mode == "ptt" else "Running - Listening...")

    def cleanup(self):
        if self.audio_recorder:
            self.audio_recorder.stop_stream()


def main():
    root = tk.Tk()
    app = VoiceChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
