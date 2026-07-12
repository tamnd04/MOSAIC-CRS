"""Lazy local speech-to-text using faster-whisper.

Audio arrives from the browser as 16 kHz, mono, signed PCM16. No cloud API or API
key is used. The selected Whisper weights are downloaded once on the first run and
then cached locally by faster-whisper/Hugging Face.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np


class LocalWhisperSTT:
    """Thread-safe, lazy wrapper around :class:`faster_whisper.WhisperModel`."""

    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = None if not language or language.lower() == "auto" else language
        self._model = None
        self._load_lock = threading.RLock()
        self._transcribe_lock = threading.RLock()
        self._error: Optional[str] = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if self._error:
                raise RuntimeError(self._error)
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                print(
                    "[local stt] loaded "
                    f"model={self.model_name} device={self.device} compute_type={self.compute_type}"
                )
            except Exception as exc:  # pragma: no cover - depends on local runtime
                self._error = f"Failed to load faster-whisper model: {exc}"
                raise RuntimeError(self._error) from exc

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe one endpointed utterance represented as little-endian PCM16."""
        if sample_rate != 16000:
            raise ValueError(f"Expected 16000 Hz audio, received {sample_rate} Hz.")
        if len(audio_bytes) < int(sample_rate * 2 * 0.18):
            return ""

        self.ensure_loaded()
        waveform = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        if waveform.size == 0:
            return ""

        with self._transcribe_lock:
            segments, _info = self._model.transcribe(
                waveform,
                language=self.language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return " ".join(text.split()).strip()

    def status(self) -> Dict[str, Any]:
        return {
            "provider": "faster-whisper",
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language or "auto",
            "loaded": self._model is not None,
            "error": self._error,
            "api_key_required": False,
        }
