"""Local wake-word detection ("Hey Jarvis") built on openWakeWord.

openWakeWord ships a pretrained ``hey_jarvis`` model that runs on CPU via
ONNX Runtime, so detection stays fully on-device. The model consumes 16 kHz
mono int16 audio in 80 ms frames (1280 samples) and emits a 0..1 score.

Requires the ``openwakeword`` package (``pip install 'OpenJarvis[voice]'``).
"""

from __future__ import annotations

import re
import threading
from collections import deque
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
    * ``"command"`` -- wake phrase and request said in one breath (keyword
      fallback only); collect the request text with :meth:`take_text`

    With a ``keyword_transcriber``, short speech segments the wake model did
    not fire on are transcribed and checked for the wake phrase. The
    pretrained model is unreliable for some accents ("Hey Javis"), while a
    small Whisper model prompted with the phrase recognizes them.
    """

    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        is_speech: Optional[Callable[[Any], bool]] = None,
        silence_threshold: float = 500.0,
        silence_seconds: float = 1.5,
        startup_silence_seconds: float = 5.0,
        max_seconds: float = 30.0,
        min_speech_seconds: float = 0.24,
        keyword_transcriber: Optional[Callable[[Any], str]] = None,
        max_keyword_seconds: float = 8.0,
        keyword_end_silence_seconds: float = 0.4,
        rewake_max_seconds: float = 3.0,
    ) -> None:
        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        self.detector = detector
        self.silence_threshold = silence_threshold
        # Loudness is a poor speech test: noise bursts pass it and quiet,
        # noise-suppressed browser speech fails it. Callers should inject a
        # VAD (see make_vad_speech_check); RMS remains the fallback.
        self._is_speech = is_speech or self._rms_is_speech
        self._silence_frames = max(1, int(silence_seconds * frames_per_second))
        self._startup_frames = max(1, int(startup_silence_seconds * frames_per_second))
        self._max_frames = max(1, int(max_seconds * frames_per_second))
        self._min_speech_frames = max(1, round(min_speech_seconds * frames_per_second))
        self._keyword_transcriber = keyword_transcriber
        self._max_keyword_frames = max(1, int(max_keyword_seconds * frames_per_second))
        self._keyword_end_frames = max(
            1, round(keyword_end_silence_seconds * frames_per_second)
        )
        self._rewake_max_frames = max(1, int(rewake_max_seconds * frames_per_second))
        self._preroll: deque = deque(maxlen=3)
        self._kw_frames: Optional[list[Any]] = None
        self._kw_speech = 0
        self._kw_silence = 0
        self._kw_too_long = False
        self._text: Optional[str] = None
        self.state = "listening"
        self._pending: Any = None
        self._recorded: list[Any] = []
        self._utterance: Optional[bytes] = None
        self._speech_frames = 0
        self._silence_count = 0

    def _rms_is_speech(self, frame: Any) -> bool:
        import numpy as np

        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        return rms > self.silence_threshold

    @property
    def _has_speech(self) -> bool:
        return self._speech_frames >= self._min_speech_frames

    def reset(self) -> None:
        """Drop any partial audio and return to listening."""
        self.state = "listening"
        self._pending = None
        self._recorded = []
        self._speech_frames = 0
        self._silence_count = 0
        self._reset_keyword()
        self.detector.reset()

    def _reset_keyword(self) -> None:
        self._kw_frames = None
        self._kw_speech = 0
        self._kw_silence = 0
        self._kw_too_long = False

    def take_text(self) -> Optional[str]:
        """Return a one-breath request's text (once), or None."""
        text, self._text = self._text, None
        return text

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
            if isinstance(event, list):
                events.extend(event)
            elif event:
                events.append(event)
        self._pending = samples[usable:]
        return events

    def _start_recording(self) -> None:
        self.detector.reset()
        self._reset_keyword()
        self.state = "recording"
        self._recorded = []
        self._speech_frames = 0
        self._silence_count = 0

    def _feed_frame(self, frame: Any) -> Any:
        import numpy as np

        if self.state == "listening":
            if self.detector.score(frame) >= self.detector.threshold:
                self._start_recording()
                return "wake"
            if self._keyword_transcriber is not None:
                return self._keyword_frame(frame)
            return None

        self._recorded.append(frame.copy())
        if self._is_speech(frame):
            self._speech_frames += 1
            self._silence_count = 0
        elif self._speech_frames:
            self._silence_count += 1
            if not self._has_speech and self._silence_count >= self._silence_frames:
                # A blip too short to be a request (a cough, a click): forget
                # it so the no-speech timeout still applies.
                self._speech_frames = 0
                self._silence_count = 0

        ended = self._has_speech and self._silence_count >= self._silence_frames
        if ended or len(self._recorded) >= self._max_frames:
            if not self._has_speech:
                self.reset()
                return "timeout"
            audio = np.concatenate(self._recorded)
            if self._is_repeated_wake_phrase(audio):
                # The speaker repeated "Hey Jarvis" (often because they were
                # unsure it heard them): wake again rather than send it.
                self._start_recording()
                return "wake"
            self._utterance = _pcm_to_wav(audio)
            self.reset()
            return "utterance"
        if not self._has_speech and len(self._recorded) >= self._startup_frames:
            self.reset()
            return "timeout"
        return None

    def _is_repeated_wake_phrase(self, audio: Any) -> bool:
        if self._keyword_transcriber is None:
            return False
        # Judge by how long they spoke, not the recording: a pause before the
        # repeat is normal ("…nothing happened? Hey Jarvis").
        if self._speech_frames > self._rewake_max_frames:
            return False
        try:
            text = self._keyword_transcriber(audio) or ""
        except Exception:
            return False
        return split_wake_phrase(text) == ""

    def _keyword_frame(self, frame: Any) -> Any:
        """Track one speech segment while listening; check it for the wake phrase."""
        import numpy as np

        speech = self._is_speech(frame)
        if self._kw_frames is None:
            if speech:
                self._kw_frames = [*self._preroll, frame.copy()]
                self._kw_speech, self._kw_silence = 1, 0
            self._preroll.append(frame.copy())
            return None

        if not self._kw_too_long:
            self._kw_frames.append(frame.copy())
            if (
                len(self._kw_frames)
                > self._max_keyword_frames + self._keyword_end_frames
            ):
                # Ordinary conversation, not a wake phrase: stop buffering but keep
                # tracking until the speaker pauses.
                self._kw_too_long = True
                self._kw_frames = []
        if speech:
            self._kw_speech += 1
            self._kw_silence = 0
        else:
            self._kw_silence += 1
        self._preroll.append(frame.copy())
        if self._kw_silence < self._keyword_end_frames:
            return None

        frames, enough = self._kw_frames, self._kw_speech >= self._min_speech_frames
        too_long = self._kw_too_long
        self._reset_keyword()
        if too_long or not enough:
            return None
        try:
            text = self._keyword_transcriber(np.concatenate(frames)) or ""
        except Exception:
            return None
        rest = split_wake_phrase(text)
        if rest is None:
            return None
        if rest:
            self._text = rest
            self.detector.reset()
            return ["wake", "command"]
        self._start_recording()
        return "wake"


# Wake-name spellings Whisper produces for "Jarvis" across accents.
_WAKE_PHRASE = re.compile(
    r"^\W*(?:(?:hey|hi|hay|okay|ok)\W+)?"
    r"(?:jarvis|javis|jarvas|jervis|jarves|jarvus|jarviz)\b\W*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def split_wake_phrase(text: str) -> Optional[str]:
    """Return the request after a leading wake phrase, "" for the phrase alone,
    or None when the text does not start with the wake phrase."""
    match = _WAKE_PHRASE.match(text or "")
    if match is None:
        return None
    return match.group(1).strip()


def make_whisper_keyword_transcriber(
    model_size: str = "base", language: Optional[str] = "en"
) -> Optional[Callable[[Any], str]]:
    """Return a fast local Whisper transcriber for wake-phrase checks, or None.

    Prompting with the wake phrase biases recognition toward "Jarvis" for
    speakers the pretrained wake model misses.
    """
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception:
        return None

    def transcribe(samples: Any) -> str:
        audio = samples.astype("float32") / 32768.0
        segments, _ = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            without_timestamps=True,
            initial_prompt="Hey Jarvis.",
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    return transcribe


def make_vad_speech_check(threshold: float = 0.5) -> Optional[Callable[[Any], bool]]:
    """Return a Silero-VAD speech test for 80 ms frames, or None if unavailable.

    openWakeWord bundles the Silero VAD ONNX model; each call returns whether
    the frame's speech probability reaches ``threshold``.
    """
    try:
        from openwakeword.vad import VAD

        vad = VAD()
    except Exception:
        return None

    def is_speech(frame: Any) -> bool:
        return float(vad.predict(frame, frame_size=640)) >= threshold

    return is_speech


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
    "make_vad_speech_check",
    "make_whisper_keyword_transcriber",
    "split_wake_phrase",
]
