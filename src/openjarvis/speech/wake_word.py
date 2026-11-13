"""Local wake-word detection ("Hey Jarvis") built on openWakeWord.

openWakeWord ships a pretrained ``hey_jarvis`` model that runs on CPU via
ONNX Runtime, so detection stays fully on-device. The model consumes 16 kHz
mono int16 audio in 80 ms frames (1280 samples) and emits a 0..1 score.

Requires the ``openwakeword`` package (``pip install 'OpenJarvis[voice]'``).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms at 16 kHz, openWakeWord's native frame size
DEFAULT_WAKE_WORD = "hey_jarvis"
DEFAULT_THRESHOLD = 0.5

_DOWNLOAD_LOCK = threading.Lock()


class WakeWordUnavailable(RuntimeError):
    """Raised when the wake-word dependencies or model cannot be loaded."""


def _load_model(wake_word: str) -> Any:
    try:
        import openwakeword.utils as oww_utils
        from openwakeword.model import Model
    except ImportError as exc:
        raise WakeWordUnavailable(
            "openwakeword is required for wake-word mode. "
            "Install with: pip install 'OpenJarvis[voice]'"
        ) from exc

    def _build() -> Any:
        return Model(wakeword_models=[wake_word], inference_framework="onnx")

    try:
        return _build()
    except Exception:
        # First run: the pretrained models are fetched on demand.
        with _DOWNLOAD_LOCK:
            try:
                oww_utils.download_models(model_names=[wake_word])
                return _build()
            except Exception as exc:
                raise WakeWordUnavailable(
                    f"Could not load wake-word model {wake_word!r}: {exc}"
                ) from exc


class WakeWordDetector:
    """Score audio frames against one wake word and report activations."""

    def __init__(
        self,
        wake_word: str = DEFAULT_WAKE_WORD,
        threshold: float = DEFAULT_THRESHOLD,
        *,
        model: Any = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"wake threshold must be in (0, 1), got {threshold}")
        self.wake_word = wake_word
        self.threshold = threshold
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _load_model(self.wake_word)
        return self._model

    def score(self, frame: Any) -> float:
        """Return the wake-word score for one int16 audio frame."""
        scores = self.model.predict(frame)
        return float(scores.get(self.wake_word, 0.0))

    def reset(self) -> None:
        """Clear the model's rolling audio buffer (e.g. after an activation)."""
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def detect_in_audio(self, samples: Any) -> bool:
        """Return True if the wake word occurs anywhere in int16 ``samples``."""
        try:
            for start in range(0, len(samples) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                if self.score(samples[start : start + FRAME_SAMPLES]) >= self.threshold:
                    return True
            return False
        finally:
            self.reset()

    def wait(
        self,
        *,
        stop: Optional[Callable[[], bool]] = None,
        stream_factory: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """Block on the microphone until the wake word is heard.

        Returns True on activation, or False if ``stop()`` became true first.
        ``stream_factory`` lets tests inject a fake input stream.
        """
        import numpy as np

        if stream_factory is None:
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise WakeWordUnavailable(
                    "sounddevice is required for wake-word mode. "
                    "Install with: pip install sounddevice"
                ) from exc

            def stream_factory() -> Any:
                return sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    blocksize=FRAME_SAMPLES,
                )

        self.model  # load before opening the mic so errors surface early
        self.reset()
        with stream_factory() as stream:
            while stop is None or not stop():
                raw, _ = stream.read(FRAME_SAMPLES)
                frame = np.frombuffer(bytes(raw), dtype=np.int16)
                if self.score(frame) >= self.threshold:
                    self.reset()
                    return True
        return False


class WakeSession:
    """Push-based wake-then-record state machine for streamed audio.

    Callers feed 16 kHz int16 frames (any length; they are re-chunked to
    :data:`FRAME_SAMPLES`). In ``listening`` state frames are scored for the
    wake word; after an activation the session records until the speaker
    pauses, mirroring ``voice_io.record_until_silence``. :meth:`feed` returns
    the events raised by the frames it consumed:

    * ``"wake"`` -- wake word heard, now recording
    * ``"utterance"`` -- recording finished; collect it with :meth:`take_wav`
    * ``"timeout"`` -- woke, but no speech followed; back to listening
    """

    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        silence_threshold: float = 500.0,
        silence_seconds: float = 1.5,
        startup_silence_seconds: float = 5.0,
        max_seconds: float = 30.0,
    ) -> None:
        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        self.detector = detector
        self.silence_threshold = silence_threshold
        self._silence_frames = max(1, int(silence_seconds * frames_per_second))
        self._startup_frames = max(1, int(startup_silence_seconds * frames_per_second))
        self._max_frames = max(1, int(max_seconds * frames_per_second))
        self.state = "listening"
        self._pending: Any = None
        self._recorded: list[Any] = []
        self._utterance: Optional[bytes] = None
        self._has_speech = False
        self._silence_count = 0

    def reset(self) -> None:
        """Drop any partial audio and return to listening."""
        self.state = "listening"
        self._pending = None
        self._recorded = []
        self._has_speech = False
        self._silence_count = 0
        self.detector.reset()

    def take_wav(self) -> Optional[bytes]:
        """Return the finished utterance as WAV bytes (once), or None."""
        wav, self._utterance = self._utterance, None
        return wav

    def feed(self, samples: Any) -> list[str]:
        import numpy as np

        samples = np.asarray(samples, dtype=np.int16)
        if self._pending is not None and len(self._pending):
            samples = np.concatenate([self._pending, samples])
        events: list[str] = []
        usable = len(samples) - len(samples) % FRAME_SAMPLES
        for start in range(0, usable, FRAME_SAMPLES):
            event = self._feed_frame(samples[start : start + FRAME_SAMPLES])
            if event:
                events.append(event)
        self._pending = samples[usable:]
        return events

    def _feed_frame(self, frame: Any) -> Optional[str]:
        import numpy as np

        if self.state == "listening":
            if self.detector.score(frame) >= self.detector.threshold:
                self.detector.reset()
                self.state = "recording"
                self._recorded = []
                self._has_speech = False
                self._silence_count = 0
                return "wake"
            return None

        self._recorded.append(frame.copy())
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms > self.silence_threshold:
            self._has_speech = True
            self._silence_count = 0
        elif not self._has_speech:
            if len(self._recorded) >= self._startup_frames:
                self.reset()
                return "timeout"
        else:
            self._silence_count += 1

        if (self._has_speech and self._silence_count >= self._silence_frames) or (
            len(self._recorded) >= self._max_frames
        ):
            if not self._has_speech:
                self.reset()
                return "timeout"
            self._utterance = _pcm_to_wav(np.concatenate(self._recorded))
            self.reset()
            return "utterance"
        return None


def _pcm_to_wav(samples: Any) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.astype("<i2").tobytes())
    return buf.getvalue()


def chime(sample_rate: int = 24000) -> None:
    """Play a short two-tone cue confirming the wake word was heard."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        return
    tones = []
    for freq in (880.0, 1320.0):
        t = np.linspace(0, 0.07, int(sample_rate * 0.07), endpoint=False)
        envelope = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
        tones.append(0.2 * envelope * np.sin(2 * np.pi * freq * t))
    sd.play(np.concatenate(tones).astype("float32"), sample_rate)
    sd.wait()


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_WAKE_WORD",
    "FRAME_SAMPLES",
    "SAMPLE_RATE",
    "WakeSession",
    "WakeWordDetector",
    "WakeWordUnavailable",
    "chime",
]
