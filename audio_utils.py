"""
Audio utilities for Voice Chat AI
Handles microphone input, VAD, and audio playback
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
from typing import Optional, Callable
import torch

import config


class AudioRecorder:
    """Handles audio recording with PTT and VAD modes"""

    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self.channels = config.CHANNELS
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stream = None

        # Level callback for UI visualization (receives float 0.0-1.0)
        self.level_callback: Optional[Callable[[float], None]] = None

        # Load Silero VAD model
        self.vad_model = None
        self._load_vad_model()

    def _load_vad_model(self):
        """Load Silero VAD model"""
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            print("VAD model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load VAD model: {e}")
            self.vad_model = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")

        # Calculate audio level for visualization
        if self.level_callback is not None:
            # Calculate RMS level
            rms = np.sqrt(np.mean(indata ** 2))
            # Normalize to 0.0-1.0 range with higher sensitivity for typical speech levels
            level = min(1.0, rms * 15.0)
            try:
                self.level_callback(level)
            except Exception:
                pass  # Ignore callback errors

        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_stream(self):
        """Start the audio input stream"""
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * config.CHUNK_DURATION)
            )
            self.stream.start()

    def stop_stream(self):
        """Stop the audio input stream"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def record_ptt(self) -> np.ndarray:
        """Record audio while key is held (push-to-talk mode)"""
        import keyboard

        self.start_stream()
        self.recording = True
        audio_chunks = []

        print(f"Hold '{config.PTT_KEY}' to speak...")

        # Wait for key press
        keyboard.wait(config.PTT_KEY)
        print("Recording... (release to stop)")

        # Record while key is held
        while keyboard.is_pressed(config.PTT_KEY):
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_chunks.append(chunk)
            except queue.Empty:
                continue

        self.recording = False

        # Clear any remaining audio in queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        if audio_chunks:
            return np.concatenate(audio_chunks, axis=0).flatten()
        return np.array([])

    def record_vad(self) -> np.ndarray:
        """Record audio using voice activity detection"""
        if self.vad_model is None:
            print("VAD model not available, falling back to PTT")
            return self.record_ptt()

        self.start_stream()
        self.recording = True
        audio_chunks = []
        speech_detected = False
        silence_start = None

        print("Listening... (speak when ready)")

        vad_iterator = self.VADIterator(
            self.vad_model,
            threshold=config.VAD_THRESHOLD,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=int(config.VAD_MIN_SILENCE_DURATION * 1000),
            speech_pad_ms=100
        )

        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                audio_chunks.append(chunk)

                # Check for speech
                chunk_tensor = torch.from_numpy(chunk.flatten())
                speech_dict = vad_iterator(chunk_tensor, return_seconds=True)

                if speech_dict:
                    if 'start' in speech_dict:
                        if not speech_detected:
                            print("Speech detected...")
                        speech_detected = True
                        silence_start = None
                    elif 'end' in speech_dict:
                        print("Speech ended")
                        break
                elif speech_detected:
                    # Track silence duration
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > config.VAD_MIN_SILENCE_DURATION:
                        print("Silence detected, stopping...")
                        break

            except queue.Empty:
                if speech_detected and silence_start and \
                   time.time() - silence_start > config.VAD_MIN_SILENCE_DURATION:
                    break
                continue
            except KeyboardInterrupt:
                break

        self.recording = False
        vad_iterator.reset_states()

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        if audio_chunks:
            return np.concatenate(audio_chunks, axis=0).flatten()
        return np.array([])


def play_audio(audio_data: np.ndarray, sample_rate: int = 24000):
    """Play audio through speakers"""
    try:
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


def list_audio_devices():
    """List available audio devices"""
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print(f"\nDefault input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")


if __name__ == "__main__":
    # Test audio devices
    list_audio_devices()

    # Test recording
    print("\nTesting PTT recording...")
    recorder = AudioRecorder()
    audio = recorder.record_ptt()
    print(f"Recorded {len(audio) / config.SAMPLE_RATE:.2f} seconds of audio")
    recorder.stop_stream()
