"""
TTS Engine Abstraction Layer
Supports multiple TTS backends: MOSS-TTS, Qwen3-TTS, and OmniVoice
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch


class TTSEngine(ABC):
    """Abstract base class for TTS engines"""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the TTS model into memory"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the TTS model and free GPU memory"""
        pass

    @abstractmethod
    def synthesize(self, text: str, voice_path: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: The text to synthesize
            voice_path: Optional path to voice reference audio for cloning

        Returns:
            Audio samples as numpy array
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return the output sample rate of this TTS engine"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of this engine"""
        pass


class MOSSTTSEngine(TTSEngine):
    """MOSS-TTS Realtime Engine"""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__(device, dtype)
        self.model = None
        self.inferencer = None
        self.codec = None
        self.tokenizer = None

    @property
    def name(self) -> str:
        return "MOSS-TTS"

    def get_sample_rate(self) -> int:
        return 24000

    def _resolve_attn_implementation(self) -> str:
        """Determine best attention implementation"""
        import importlib.util
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

    def load(self) -> None:
        if self.loaded:
            return

        # Add MOSS-TTS to path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(base_dir, 'MOSS-TTS'))
        sys.path.insert(0, os.path.join(base_dir, 'MOSS-TTS', 'moss_tts_realtime'))

        from transformers import AutoModel, AutoTokenizer
        from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
        from inferencer import MossTTSRealtimeInference

        model_path = "OpenMOSS-Team/MOSS-TTS-Realtime"
        codec_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

        attn_impl = self._resolve_attn_implementation()

        # Load TTS model
        self.model = MossTTSRealtime.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
            torch_dtype=self.dtype
        ).to(self.device).eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load codec
        self.codec = AutoModel.from_pretrained(
            codec_path,
            trust_remote_code=True
        ).eval().to(self.device)

        # Create inferencer
        self.inferencer = MossTTSRealtimeInference(
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=2048,
            codec=self.codec,
            codec_sample_rate=24000,
            codec_encode_kwargs={"chunk_duration": 8}
        )

        # Patch _load_audio to use soundfile
        import torchaudio
        import soundfile as sf

        def _load_audio_soundfile(audio_path: str, target_sample_rate: int) -> torch.Tensor:
            audio_data, sr = sf.read(audio_path)
            wav = torch.from_numpy(audio_data).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim == 2 and wav.shape[1] < wav.shape[0]:
                wav = wav.T
            if sr != target_sample_rate:
                wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav

        self.inferencer._load_audio = lambda audio_path, target_sample_rate: _load_audio_soundfile(audio_path, target_sample_rate)
        self.loaded = True

    def unload(self) -> None:
        """Unload MOSS-TTS model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.inferencer is not None:
            del self.inferencer
            self.inferencer = None
        if self.codec is not None:
            del self.codec
            self.codec = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.loaded = False

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc
        gc.collect()
        print("[DEBUG] MOSS-TTS unloaded, GPU memory freed")

    def synthesize(self, text: str, voice_path: Optional[str] = None) -> np.ndarray:
        self.load()

        result = self.inferencer.generate(
            text=text,
            reference_audio_path=voice_path,
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
            decode_result = self.codec.decode(output.permute(1, 0), chunk_duration=8)
            wav = decode_result["audio"][0].cpu().detach()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            audio_samples.append(wav)

        if audio_samples:
            audio = torch.cat(audio_samples, dim=-1)
            return audio.numpy().flatten()
        return np.array([])


class Qwen3TTSEngine(TTSEngine):
    """Qwen3-TTS Engine - faster inference with built-in speakers"""

    # Available speakers for Qwen3-TTS CustomVoice model
    SPEAKERS = [
        "aiden",
        "dylan",
        "eric",
        "ono_anna",
        "ryan",
        "serena",
        "sohee",
        "uncle_fu",
        "vivian"
    ]

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 speaker: str = "serena", model_size: str = "0.6B"):
        super().__init__(device, dtype)
        self.model = None
        self.speaker = speaker
        self.model_size = model_size  # "0.6B" is faster, "1.7B" is higher quality
        self._sample_rate = 24000  # Qwen3-TTS outputs at 24kHz

    @property
    def name(self) -> str:
        return "Qwen3-TTS"

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        if self.loaded:
            return

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "Qwen3-TTS not installed. Install with: pip install -U qwen-tts"
            )

        # Determine attention implementation
        attn_impl = "sdpa"
        if self.device == "cuda":
            try:
                import importlib.util
                if importlib.util.find_spec("flash_attn") is not None:
                    major, _ = torch.cuda.get_device_capability()
                    if major >= 8:
                        attn_impl = "flash_attention_2"
            except:
                pass

        # Select model based on size
        model_name = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-CustomVoice"
        print(f"[DEBUG] Loading Qwen3-TTS model: {model_name}")
        print(f"[DEBUG] Device: {self.device}, dtype: {self.dtype}, attn: {attn_impl}")

        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
        )

        print(f"[DEBUG] Qwen3-TTS model loaded successfully")
        self.loaded = True

    def unload(self) -> None:
        """Unload Qwen3-TTS model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        self.loaded = False

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc
        gc.collect()
        print("[DEBUG] Qwen3-TTS unloaded, GPU memory freed")

    def synthesize(self, text: str, voice_path: Optional[str] = None) -> np.ndarray:
        self.load()

        if voice_path and os.path.exists(voice_path):
            # Use voice cloning if reference audio provided
            # First, try to get transcript from accompanying .txt file
            txt_path = os.path.splitext(voice_path)[0] + ".txt"
            transcript = None
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()

            if transcript:
                result = self.model.generate_voice_clone(
                    text=text,
                    language="english",
                    reference_audio=voice_path,
                    reference_transcript=transcript
                )
            else:
                # Fallback to custom voice without cloning
                result = self.model.generate_custom_voice(
                    text=text,
                    language="english",
                    speaker=self.speaker
                )
        else:
            # Use built-in speaker
            result = self.model.generate_custom_voice(
                text=text,
                language="english",
                speaker=self.speaker
            )

        # generate_custom_voice returns Tuple[List[np.ndarray], int]
        # Unpack the tuple: (audio_list, sample_rate)
        if isinstance(result, tuple):
            audio_list, returned_sample_rate = result
            self._sample_rate = returned_sample_rate  # Update sample rate from model
            print(f"[DEBUG] Qwen3-TTS returned {len(audio_list)} audio chunks, sample_rate={returned_sample_rate}")
            # Concatenate all audio chunks
            if audio_list and len(audio_list) > 0:
                audio = np.concatenate(audio_list) if len(audio_list) > 1 else audio_list[0]
                print(f"[DEBUG] Audio shape: {audio.shape}, dtype: {audio.dtype}, min: {audio.min():.4f}, max: {audio.max():.4f}")
            else:
                print("[DEBUG] No audio chunks returned!")
                return np.array([])
        else:
            audio = result

        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.flatten()

        # Normalize audio to [-1, 1] range for proper playback
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # Normalize to 95% to avoid clipping

        print(f"[DEBUG] Final audio: {len(audio)} samples, {len(audio)/self._sample_rate:.2f}s duration")
        print(f"[DEBUG] Normalized audio range: [{audio.min():.4f}, {audio.max():.4f}]")
        return audio.astype(np.float32)

    def set_speaker(self, speaker: str) -> None:
        """Change the speaker voice"""
        if speaker in self.SPEAKERS:
            self.speaker = speaker
        else:
            raise ValueError(f"Unknown speaker: {speaker}. Available: {self.SPEAKERS}")


class OmniVoiceEngine(TTSEngine):
    """OmniVoice Engine - multilingual zero-shot TTS with voice cloning and voice design (600+ languages)"""

    COMMON_LANGUAGES = [
        "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar",
        "it", "nl", "pl", "tr", "vi", "th", "id", "hi", "sv", "cs"
    ]

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 language: str = "en", voice_design: str = "",
                 num_step: int = 16, guidance_scale: float = 2.0, speed: float = 1.0,
                 voice_path: str = None):
        super().__init__(device, dtype)
        self.model = None
        self.language = language
        self.voice_design = voice_design
        self.num_step = num_step
        self.guidance_scale = guidance_scale
        self.speed = speed
        self.voice_path = voice_path  # Pre-set voice for cloning at load time
        self._cached_voice_path = None
        self._cached_clone_prompt = None

    @property
    def name(self) -> str:
        return "OmniVoice"

    def get_sample_rate(self) -> int:
        # Try to get from model config, fallback to 24000
        if self.model is not None:
            for attr in ['sample_rate', 'sampling_rate', 'audio_sample_rate']:
                sr = getattr(self.model, attr, None) or getattr(getattr(self.model, 'config', None), attr, None)
                if sr:
                    return int(sr)
        return 24000

    def load(self) -> None:
        if self.loaded:
            return

        # Patch torchaudio to use soundfile backend instead of torchcodec (avoids FFmpeg dependency)
        self._patch_torchaudio()

        try:
            from omnivoice import OmniVoice
        except ImportError:
            raise ImportError(
                "OmniVoice not installed. Install with: pip install omnivoice"
            )
        print("[DEBUG] Loading OmniVoice model...")
        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")

        # Move model to GPU if available
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.model = self.model.to(self.device)
                print(f"[DEBUG] OmniVoice moved to {self.device}")
            except Exception as e:
                print(f"[DEBUG] OmniVoice: Could not move to {self.device}: {e}")

        # Log device info
        try:
            params = next(self.model.parameters())
            print(f"[DEBUG] OmniVoice model device: {params.device}, dtype: {params.dtype}")
        except StopIteration:
            pass

        self.loaded = True
        print("[DEBUG] OmniVoice loaded")

        # Pre-create voice clone prompt at startup so synthesis is fast
        if self.voice_path and os.path.exists(self.voice_path):
            import time as _time
            print(f"[DEBUG] OmniVoice: Pre-creating voice clone prompt from {self.voice_path}...")
            t = _time.time()
            self._cached_clone_prompt = self.model.create_voice_clone_prompt(self.voice_path)
            self._cached_voice_path = self.voice_path
            print(f"[DEBUG] OmniVoice: Clone prompt ready in {_time.time()-t:.1f}s")

    @staticmethod
    def _patch_torchaudio():
        """Block torchcodec and patch torchaudio to use soundfile, avoiding FFmpeg dependency"""
        import sys
        import types
        import soundfile as sf
        import numpy as _np

        # Block torchcodec by injecting dummy modules into sys.modules BEFORE torchaudio imports it
        if 'torchcodec' not in sys.modules or not hasattr(sys.modules.get('torchcodec', None), '_is_real'):
            # Create dummy AudioDecoder class that torchaudio expects to import
            class _DummyAudioDecoder:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("torchcodec disabled - using soundfile backend instead")

            class _DummyMetadata:
                pass

            dummy = types.ModuleType('torchcodec')
            dummy.decoders = types.ModuleType('torchcodec.decoders')
            dummy.decoders.AudioDecoder = _DummyAudioDecoder
            dummy._core = types.ModuleType('torchcodec._core')
            dummy._core.AudioStreamMetadata = _DummyMetadata
            dummy._core.VideoStreamMetadata = _DummyMetadata
            sys.modules['torchcodec'] = dummy
            sys.modules['torchcodec.decoders'] = dummy.decoders
            sys.modules['torchcodec._core'] = dummy._core
            print("[DEBUG] OmniVoice: Blocked torchcodec with dummy modules")

        import torchaudio

        # Check if already patched
        if getattr(torchaudio, '_sf_patched', False):
            return

        def _sf_load(filepath, *args, **kwargs):
            """Load audio using soundfile instead of torchaudio/torchcodec"""
            audio_data, sample_rate = sf.read(str(filepath))
            if len(audio_data.shape) == 1:
                audio_data = audio_data[_np.newaxis, :]  # (1, samples)
            else:
                audio_data = audio_data.T  # (channels, samples)
            return torch.from_numpy(audio_data.astype(_np.float32)), sample_rate

        def _sf_save(filepath, waveform, sample_rate, *args, **kwargs):
            """Save audio using soundfile instead of torchaudio/torchcodec"""
            audio_np = waveform.cpu().numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T  # (samples, channels)
            sf.write(str(filepath), audio_np, int(sample_rate), subtype='PCM_16')

        def _sf_info(filepath, *args, **kwargs):
            """Get audio info using soundfile"""
            info = sf.info(str(filepath))
            class AudioInfo:
                def __init__(self, i):
                    self.sample_rate = i.samplerate
                    self.num_frames = i.frames
                    self.num_channels = i.channels
            return AudioInfo(info)

        torchaudio.load = _sf_load
        torchaudio.save = _sf_save
        torchaudio.info = _sf_info
        torchaudio._sf_patched = True

        # Also patch functional.resample to use torch interpolation
        if hasattr(torchaudio, 'functional'):
            _orig_resample = getattr(torchaudio.functional, 'resample', None)
            def _torch_resample(waveform, orig_freq, new_freq, **kwargs):
                if orig_freq == new_freq:
                    return waveform
                import torch.nn.functional as F
                new_len = int(waveform.shape[-1] * new_freq / orig_freq)
                if waveform.dim() == 1:
                    return F.interpolate(waveform.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear', align_corners=False).squeeze(0).squeeze(0)
                elif waveform.dim() == 2:
                    return F.interpolate(waveform.unsqueeze(0), size=new_len, mode='linear', align_corners=False).squeeze(0)
                return F.interpolate(waveform, size=new_len, mode='linear', align_corners=False)
            torchaudio.functional.resample = _torch_resample

        print("[DEBUG] OmniVoice: Patched torchaudio to use soundfile backend")

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self._cached_clone_prompt = None
        self._cached_voice_path = None
        self.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        print("[DEBUG] OmniVoice unloaded, GPU memory freed")

    def synthesize(self, text: str, voice_path: Optional[str] = None) -> np.ndarray:
        import time as _time
        self.load()

        generate_kwargs = {
            "num_step": self.num_step,
            "guidance_scale": self.guidance_scale,
            "speed": self.speed,
        }

        print(f"[DEBUG] OmniVoice.synthesize: text='{text[:60]}...', language={self.language}")
        print(f"[DEBUG] OmniVoice.synthesize: voice_path={voice_path}, voice_design='{self.voice_design[:30] if self.voice_design else ''}'")
        print(f"[DEBUG] OmniVoice.synthesize: num_step={self.num_step}, guidance_scale={self.guidance_scale}, speed={self.speed}")

        t_start = _time.time()

        if voice_path and os.path.exists(voice_path):
            # Voice cloning - cache the clone prompt for the same voice_path
            if voice_path != self._cached_voice_path or self._cached_clone_prompt is None:
                print(f"[DEBUG] OmniVoice: Creating voice clone prompt from {voice_path}...")
                t_clone = _time.time()
                self._cached_clone_prompt = self.model.create_voice_clone_prompt(voice_path)
                self._cached_voice_path = voice_path
                print(f"[DEBUG] OmniVoice: Clone prompt created in {_time.time()-t_clone:.2f}s")
                print(f"[DEBUG] OmniVoice: Clone prompt type={type(self._cached_clone_prompt)}")
                if hasattr(self._cached_clone_prompt, '__len__'):
                    print(f"[DEBUG] OmniVoice: Clone prompt len={len(self._cached_clone_prompt)}")
                if hasattr(self._cached_clone_prompt, 'keys'):
                    print(f"[DEBUG] OmniVoice: Clone prompt keys={list(self._cached_clone_prompt.keys())}")
            else:
                print(f"[DEBUG] OmniVoice: Using cached clone prompt")
            print(f"[DEBUG] OmniVoice: Generating with voice cloning (prompt type={type(self._cached_clone_prompt).__name__})...")
            audio = self.model.generate(
                text, language=self.language, voice_clone_prompt=self._cached_clone_prompt,
                **generate_kwargs
            )
        elif self.voice_design:
            print(f"[DEBUG] OmniVoice: Generating with voice design: '{self.voice_design}'...")
            audio = self.model.generate(
                text, language=self.language, instruct=self.voice_design,
                **generate_kwargs
            )
        else:
            print(f"[DEBUG] OmniVoice: Generating with default voice...")
            audio = self.model.generate(
                text, language=self.language, **generate_kwargs
            )

        t_gen = _time.time() - t_start
        print(f"[DEBUG] OmniVoice: Generation completed in {t_gen:.2f}s")
        print(f"[DEBUG] OmniVoice: Output type={type(audio)}, shape/len={getattr(audio, 'shape', None) or (len(audio) if hasattr(audio, '__len__') else 'unknown')}")

        # Convert output to 1D float32 numpy array
        if isinstance(audio, torch.Tensor):
            print(f"[DEBUG] OmniVoice: Tensor output, shape={audio.shape}, dtype={audio.dtype}")
            audio_np = audio.squeeze().cpu().float().numpy()
        elif isinstance(audio, tuple):
            print(f"[DEBUG] OmniVoice: Tuple output, len={len(audio)}, types={[type(x).__name__ for x in audio]}")
            item = audio[0]
            if isinstance(item, torch.Tensor):
                audio_np = item.squeeze().cpu().float().numpy()
            else:
                audio_np = np.array(item).flatten()
        elif isinstance(audio, list):
            print(f"[DEBUG] OmniVoice: List output, len={len(audio)}")
            if len(audio) > 0 and isinstance(audio[0], torch.Tensor):
                print(f"[DEBUG] OmniVoice: List[0] shape={audio[0].shape}, dtype={audio[0].dtype}")
                audio_np = torch.cat([a.squeeze() for a in audio]).cpu().float().numpy()
            else:
                audio_np = np.concatenate([np.array(a).flatten() for a in audio])
        else:
            audio_np = np.array(audio).flatten()

        if audio_np.ndim > 1:
            print(f"[DEBUG] OmniVoice: Flattening from shape {audio_np.shape}")
            audio_np = audio_np.flatten()

        # Normalize to prevent clipping
        max_val = np.abs(audio_np).max()
        print(f"[DEBUG] OmniVoice: Pre-normalize max={max_val:.4f}, min={audio_np.min():.4f}")
        if max_val > 0:
            audio_np = audio_np / max_val * 0.95

        sr = self.get_sample_rate()
        print(f"[DEBUG] OmniVoice: Final audio: {len(audio_np)} samples, {len(audio_np)/sr:.2f}s at {sr}Hz")
        return audio_np.astype(np.float32)


# Factory function to create TTS engines
def create_tts_engine(engine_name: str, device: str = "cuda",
                       dtype: torch.dtype = torch.bfloat16,
                       **kwargs) -> TTSEngine:
    """
    Create a TTS engine by name.

    Args:
        engine_name: "MOSS-TTS", "Qwen3-TTS", or "OmniVoice"
        device: Device to use ("cuda" or "cpu")
        dtype: Data type for model
        **kwargs: Additional arguments for specific engines

    Returns:
        TTSEngine instance
    """
    engines = {
        "MOSS-TTS": MOSSTTSEngine,
        "Qwen3-TTS": Qwen3TTSEngine,
        "OmniVoice": OmniVoiceEngine,
    }

    if engine_name not in engines:
        raise ValueError(f"Unknown TTS engine: {engine_name}. Available: {list(engines.keys())}")

    return engines[engine_name](device=device, dtype=dtype, **kwargs)


# Available TTS engines for UI
TTS_ENGINE_OPTIONS = ["MOSS-TTS", "Qwen3-TTS", "OmniVoice"]
