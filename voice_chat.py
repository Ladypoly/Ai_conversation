"""
Voice Chat AI - Main Application
Integrates WhisperX (STT), Qwen3.5 (LLM via Ollama), and MOSS-TTS (TTS)
"""

import sys
import os
import importlib.util
import threading
import time
import queue
import re
import numpy as np
import torch
import torchaudio
import requests
import json
import keyboard
import soundfile as sf
import tempfile

# Add MOSS-TTS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MOSS-TTS'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MOSS-TTS', 'moss_tts_realtime'))

import config

# Override config with runtime settings if available
try:
    import runtime_config
    if hasattr(runtime_config, 'SYSTEM_PROMPT'):
        config.SYSTEM_PROMPT = runtime_config.SYSTEM_PROMPT
    if hasattr(runtime_config, 'DEFAULT_INPUT_MODE'):
        config.DEFAULT_INPUT_MODE = runtime_config.DEFAULT_INPUT_MODE
    if hasattr(runtime_config, 'WHISPER_MODEL'):
        config.WHISPER_MODEL = runtime_config.WHISPER_MODEL
except ImportError:
    pass

from audio_utils import AudioRecorder, play_audio, list_audio_devices


class VoiceChatAI:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.input_mode = config.DEFAULT_INPUT_MODE
        self.streaming_mode = True  # Enable streaming LLM->TTS by default
        self.running = False
        self.conversation_history = []

        # Models (loaded lazily)
        self.whisper_model = None
        self.tts_model = None
        self.tts_inferencer = None
        self.tts_codec = None
        self.audio_recorder = None

        print(f"Using device: {self.device}")

    def load_whisper(self):
        """Load WhisperX model"""
        if self.whisper_model is None:
            print("Loading WhisperX model...")
            import whisperx
            self.whisper_model = whisperx.load_model(
                config.WHISPER_MODEL,
                self.device,
                compute_type=config.WHISPER_COMPUTE_TYPE
            )
            print("WhisperX loaded")

    def _resolve_attn_implementation(self) -> str:
        """Determine best attention implementation"""
        if (
            self.device == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and self.dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        if self.device == "cuda":
            return "sdpa"
        return "eager"

    def load_tts(self):
        """Load MOSS-TTS Realtime model and codec"""
        if self.tts_model is None:
            print("Loading MOSS-TTS Realtime model...")
            from transformers import AutoModel, AutoTokenizer
            from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
            from inferencer import MossTTSRealtimeInference

            model_path = config.TTS_MODEL
            codec_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

            attn_impl = self._resolve_attn_implementation()
            print(f"Using attention implementation: {attn_impl}")

            # Load TTS model
            self.tts_model = MossTTSRealtime.from_pretrained(
                model_path,
                attn_implementation=attn_impl,
                torch_dtype=self.dtype
            ).to(self.device)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load codec
            print("Loading MOSS Audio Tokenizer...")
            self.tts_codec = AutoModel.from_pretrained(
                codec_path,
                trust_remote_code=True
            ).eval().to(self.device)

            # Create inferencer
            self.tts_inferencer = MossTTSRealtimeInference(
                model=self.tts_model,
                tokenizer=tokenizer,
                max_length=5000,
                codec=self.tts_codec,
                codec_sample_rate=24000,
                codec_encode_kwargs={"chunk_duration": 8}
            )

            # Patch _load_audio to use soundfile instead of torchaudio
            # (torchaudio uses torchcodec which requires FFmpeg shared libs)
            def _load_audio_soundfile(audio_path: str, target_sample_rate: int) -> torch.Tensor:
                import soundfile as sf_load
                audio_data, sr = sf_load.read(audio_path)
                # Convert to torch tensor (soundfile returns numpy)
                wav = torch.from_numpy(audio_data).float()
                # Ensure shape is (channels, samples)
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                elif wav.ndim == 2 and wav.shape[1] < wav.shape[0]:
                    # (samples, channels) -> (channels, samples)
                    wav = wav.T
                # Resample if needed
                if sr != target_sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
                # Convert to mono if stereo
                if wav.ndim == 2 and wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                return wav

            # Monkey-patch the method to use soundfile
            self.tts_inferencer._load_audio = lambda audio_path, target_sample_rate: _load_audio_soundfile(audio_path, target_sample_rate)
            print("MOSS-TTS Realtime loaded (with soundfile audio loading)")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using WhisperX"""
        self.load_whisper()

        # Save audio to temp file (WhisperX needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, config.SAMPLE_RATE)
            temp_path = f.name

        try:
            result = self.whisper_model.transcribe(temp_path, batch_size=16)
            text = " ".join([seg["text"] for seg in result["segments"]])
            return text.strip()
        finally:
            os.unlink(temp_path)

    def chat_ollama(self, user_message: str) -> str:
        """Send message to Ollama and get response"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        messages.extend(self.conversation_history[-10:])  # Keep last 10 turns

        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            assistant_message = result["message"]["content"]

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message
        except requests.exceptions.RequestException as e:
            print(f"Ollama error: {e}")
            return "Sorry, I couldn't connect to the language model."

    def synthesize_speech(self, text: str) -> np.ndarray:
        """Synthesize speech using MOSS-TTS Realtime"""
        self.load_tts()

        # Generate audio tokens
        result = self.tts_inferencer.generate(
            text=text,
            reference_audio_path=None,  # No voice cloning, use default voice
            temperature=0.8,
            top_p=0.6,
            top_k=30,
            repetition_penalty=1.1,
            repetition_window=50,
            device=self.device,
        )

        # Decode audio from tokens
        audio_samples = []
        for generated_tokens in result:
            output = torch.tensor(generated_tokens).to(self.device)
            decode_result = self.tts_codec.decode(output.permute(1, 0), chunk_duration=8)
            wav = decode_result["audio"][0].cpu().detach()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            audio_samples.append(wav)

        if audio_samples:
            audio = torch.cat(audio_samples, dim=-1)
            return audio.numpy().flatten()
        return np.array([])

    def stream_chat_ollama(self, user_message: str):
        """Stream LLM response and yield text chunks at sentence boundaries"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        messages.extend(self.conversation_history[-10:])

        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
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
            min_chunk_len = 50  # Minimum characters for a chunk

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            buffer += token
                            full_response += token

                            # Check for sentence boundaries
                            while True:
                                match = split_pattern.search(buffer)
                                if match:
                                    end_pos = match.end()
                                    chunk = buffer[:end_pos].strip()

                                    # Require minimum length to avoid tiny chunks
                                    if len(chunk) < min_chunk_len:
                                        break

                                    if chunk:
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
                        yield chunk
                    buffer = buffer[end_pos:]
                else:
                    # No more valid split points, yield what's left
                    if buffer.strip():
                        yield buffer.strip()
                    break

            # Save full response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })

        except requests.exceptions.RequestException as e:
            print(f"Ollama error: {e}")
            yield "Sorry, I couldn't connect to the language model."

    def process_turn_streaming(self):
        """Process one conversation turn with streaming LLM->TTS"""
        # Record audio
        if self.input_mode == "ptt":
            audio = self.audio_recorder.record_ptt()
        else:
            audio = self.audio_recorder.record_vad()

        if len(audio) < config.SAMPLE_RATE * 0.5:
            print("Audio too short, skipping...")
            return

        # Transcribe
        print("Transcribing...")
        user_text = self.transcribe(audio)
        if not user_text:
            print("No speech detected")
            return

        print(f"\nYou: {user_text}")

        # Stream LLM and TTS with overlap
        print("Assistant: ", end="", flush=True)

        self.load_tts()  # Ensure TTS is loaded

        audio_queue = queue.Queue()
        playback_done = threading.Event()

        def audio_player():
            """Background thread to play audio chunks"""
            import sounddevice as sd
            while True:
                audio_chunk = audio_queue.get()
                if audio_chunk is None:  # Sentinel to stop
                    break
                try:
                    sd.play(audio_chunk, 24000)
                    sd.wait()
                except Exception as e:
                    print(f"\nPlayback error: {e}")
            playback_done.set()

        # Start audio player thread
        player_thread = threading.Thread(target=audio_player, daemon=True)
        player_thread.start()

        full_response = ""
        chunk_count = 0

        # Set a consistent seed for this turn so all chunks use the same voice
        turn_seed = int(time.time()) % 10000

        for text_chunk in self.stream_chat_ollama(user_text):
            print(text_chunk, end=" ", flush=True)
            full_response += text_chunk + " "

            # Generate TTS for this chunk
            try:
                # Reset seed before each TTS call to get consistent voice
                torch.manual_seed(turn_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(turn_seed)

                audio_chunk = self.synthesize_speech(text_chunk)
                if len(audio_chunk) > 0:
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
            except Exception as e:
                print(f"\nTTS error: {e}")

        print()  # New line after response

        # Signal player to stop and wait
        audio_queue.put(None)
        playback_done.wait(timeout=30)

    def toggle_input_mode(self):
        """Toggle between PTT and VAD modes"""
        if self.input_mode == "ptt":
            self.input_mode = "vad"
            print("\nSwitched to Voice Activity Detection (VAD) mode")
        else:
            self.input_mode = "ptt"
            print("\nSwitched to Push-to-Talk (PTT) mode")

    def toggle_streaming_mode(self):
        """Toggle between streaming and non-streaming mode"""
        self.streaming_mode = not self.streaming_mode
        mode = "STREAMING" if self.streaming_mode else "BATCH"
        print(f"\nSwitched to {mode} mode")

    def process_turn(self):
        """Process one conversation turn"""
        # Record audio
        if self.input_mode == "ptt":
            audio = self.audio_recorder.record_ptt()
        else:
            audio = self.audio_recorder.record_vad()

        if len(audio) < config.SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
            print("Audio too short, skipping...")
            return

        # Transcribe
        print("Transcribing...")
        user_text = self.transcribe(audio)
        if not user_text:
            print("No speech detected")
            return

        print(f"\nYou: {user_text}")

        # Get LLM response
        print("Thinking...")
        response_text = self.chat_ollama(user_text)
        print(f"\nAssistant: {response_text}")

        # Synthesize and play response
        print("Generating speech...")
        audio_response = self.synthesize_speech(response_text)
        play_audio(audio_response, sample_rate=24000)

    def run(self):
        """Main conversation loop"""
        print("\n" + "="*50)
        print("Voice Chat AI (Streaming Mode)")
        print("="*50)
        print(f"\nModels:")
        print(f"  - STT: WhisperX ({config.WHISPER_MODEL})")
        print(f"  - LLM: {config.OLLAMA_MODEL} via Ollama")
        print(f"  - TTS: MOSS-TTS Realtime")
        print(f"\nModes:")
        print(f"  - Input: {self.input_mode.upper()}")
        print(f"  - Response: {'STREAMING' if self.streaming_mode else 'BATCH'}")
        print(f"\nControls:")
        print(f"  - {config.PTT_KEY.upper()}: Hold to speak (PTT mode)")
        print(f"  - {config.MODE_TOGGLE_KEY.upper()}: Toggle PTT/VAD mode")
        print(f"  - S: Toggle streaming mode")
        print(f"  - ESC: Exit")
        print("="*50 + "\n")

        # Initialize audio recorder
        self.audio_recorder = AudioRecorder()

        # Set up keyboard shortcuts
        keyboard.on_press_key(config.MODE_TOGGLE_KEY, lambda _: self.toggle_input_mode())
        keyboard.on_press_key('s', lambda _: self.toggle_streaming_mode())

        self.running = True

        try:
            while self.running:
                if keyboard.is_pressed('escape'):
                    print("\nExiting...")
                    break

                # Use streaming or batch mode
                if self.streaming_mode:
                    self.process_turn_streaming()
                else:
                    self.process_turn()
                print("\n" + "-"*30 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.running = False
            if self.audio_recorder:
                self.audio_recorder.stop_stream()
            print("Goodbye!")


def main():
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check Ollama
    try:
        response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        print("Ollama is running")
    except Exception as e:
        print(f"Warning: Ollama may not be running: {e}")
        print(f"Start Ollama with: ollama serve")

    # List audio devices
    list_audio_devices()

    # Run voice chat
    chat = VoiceChatAI()
    chat.run()


if __name__ == "__main__":
    main()
