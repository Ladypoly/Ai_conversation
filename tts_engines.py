"""
TTS Engine Abstraction Layer
Supports multiple TTS backends: MOSS-TTS and Qwen3-TTS
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

        print(f"[DEBUG] Final audio: {len(audio)} samples, {len(audio)/self._sample_rate:.2f}s duration")
        return audio

    def set_speaker(self, speaker: str) -> None:
        """Change the speaker voice"""
        if speaker in self.SPEAKERS:
            self.speaker = speaker
        else:
            raise ValueError(f"Unknown speaker: {speaker}. Available: {self.SPEAKERS}")


# Factory function to create TTS engines
def create_tts_engine(engine_name: str, device: str = "cuda",
                       dtype: torch.dtype = torch.bfloat16,
                       **kwargs) -> TTSEngine:
    """
    Create a TTS engine by name.

    Args:
        engine_name: "MOSS-TTS" or "Qwen3-TTS"
        device: Device to use ("cuda" or "cpu")
        dtype: Data type for model
        **kwargs: Additional arguments for specific engines

    Returns:
        TTSEngine instance
    """
    engines = {
        "MOSS-TTS": MOSSTTSEngine,
        "Qwen3-TTS": Qwen3TTSEngine,
    }

    if engine_name not in engines:
        raise ValueError(f"Unknown TTS engine: {engine_name}. Available: {list(engines.keys())}")

    return engines[engine_name](device=device, dtype=dtype, **kwargs)


# Available TTS engines for UI
TTS_ENGINE_OPTIONS = ["MOSS-TTS", "Qwen3-TTS"]
