"""Tests for the "Hey Jarvis" wake-word detector and chat integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openjarvis.speech.wake_word import FRAME_SAMPLES, WakeWordDetector


class FakeModel:
    """Scores a frame 0.9 if its first sample is the marker value 7, else 0."""

    def __init__(self) -> None:
        self.resets = 0
        self.frames = 0

    def predict(self, frame):
        self.frames += 1
        return {"hey_jarvis": 0.9 if int(frame[0]) == 7 else 0.0}

    def reset(self) -> None:
        self.resets += 1


class FakeStream:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        assert n == FRAME_SAMPLES
        frame = self._frames.pop(0) if self._frames else np.zeros(n, np.int16)
        return frame.tobytes(), False


def _frame(marker: int = 0) -> np.ndarray:
    frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    frame[0] = marker
    return frame


def test_detect_in_audio_finds_marker_and_resets():
    model = FakeModel()
    det = WakeWordDetector(model=model)
    audio = np.concatenate([_frame(), _frame(), _frame(7), _frame()])
    assert det.detect_in_audio(audio) is True
    assert model.resets == 1


def test_detect_in_audio_rejects_silence():
    det = WakeWordDetector(model=FakeModel())
    assert det.detect_in_audio(np.zeros(FRAME_SAMPLES * 5, np.int16)) is False


def test_threshold_is_respected():
    det = WakeWordDetector(threshold=0.95, model=FakeModel())
    assert det.detect_in_audio(_frame(7)) is False


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        WakeWordDetector(threshold=threshold, model=FakeModel())


def test_wait_returns_true_on_activation():
    model = FakeModel()
    det = WakeWordDetector(model=model)
    frames = [_frame(), _frame(), _frame(7)]
    assert det.wait(stream_factory=lambda: FakeStream(frames)) is True
    assert model.frames == 3


def test_wait_returns_false_when_stopped():
    det = WakeWordDetector(model=FakeModel())
    calls = iter([False, False, True])
    assert (
        det.wait(stop=lambda: next(calls), stream_factory=lambda: FakeStream([]))
        is False
    )


def test_session_builds_detector_from_config():
    from openjarvis.cli._voice_chat import VoiceSession

    config = SimpleNamespace(
        speech=SimpleNamespace(wake_word="hey_jarvis", wake_threshold=0.7)
    )
    session = VoiceSession(config)
    det = session.get_wake_detector()
    assert det.wake_word == "hey_jarvis"
    assert det.threshold == 0.7
    assert session.get_wake_detector() is det


def test_wait_for_wake_word_records_after_activation(monkeypatch):
    from openjarvis.cli import _voice_chat

    monkeypatch.setattr("openjarvis.speech.wake_word.chime", lambda: None)
    monkeypatch.setattr(_voice_chat, "record_voice", lambda c, s: "what time is it")
    det = SimpleNamespace(wake_word="hey_jarvis", wait=lambda: True)
    console = SimpleNamespace(print=lambda *a, **k: None)
    result = _voice_chat.wait_for_wake_word(console, object(), det)
    assert result == "what time is it"


def test_wait_for_wake_word_ctrl_c_exits(monkeypatch):
    from openjarvis.cli import _voice_chat

    def interrupted():
        raise KeyboardInterrupt

    det = SimpleNamespace(wake_word="hey_jarvis", wait=interrupted)
    console = SimpleNamespace(print=lambda *a, **k: None)
    assert (
        _voice_chat.wait_for_wake_word(console, object(), det) is _voice_chat.VOICE_EXIT
    )


def test_chat_exposes_wake_word_flag():
    from click.testing import CliRunner

    from openjarvis.cli.chat_cmd import chat

    result = CliRunner().invoke(chat, ["--help"])
    assert "--wake-word" in result.output


# --- WakeSession: streamed wake -> record -> utterance state machine ---------


def _loud(n_frames: int = 1) -> np.ndarray:
    return np.full(FRAME_SAMPLES * n_frames, 3000, dtype=np.int16)


def _quiet(n_frames: int = 1) -> np.ndarray:
    return np.zeros(FRAME_SAMPLES * n_frames, dtype=np.int16)


def _session(**kwargs):
    from openjarvis.speech.wake_word import WakeSession

    return WakeSession(WakeWordDetector(model=FakeModel()), **kwargs)


def test_session_wake_then_utterance_on_silence():
    session = _session(silence_seconds=0.24)  # 3 frames
    assert session.feed(_quiet(2)) == []
    assert session.feed(_frame(7)) == ["wake"]
    assert session.state == "recording"
    assert session.feed(_loud(5)) == []
    assert session.feed(_quiet(3)) == ["utterance"]
    assert session.state == "listening"
    wav = session.take_wav()
    assert wav[:4] == b"RIFF"
    assert session.take_wav() is None


def test_session_rechunks_partial_frames():
    session = _session()
    frame = _frame(7)
    assert session.feed(frame[:500]) == []
    assert session.feed(frame[500:]) == ["wake"]


def test_session_times_out_without_speech():
    session = _session(startup_silence_seconds=0.4)  # 5 frames
    assert session.feed(_frame(7)) == ["wake"]
    assert session.feed(_quiet(5)) == ["timeout"]
    assert session.state == "listening"
    assert session.take_wav() is None


def test_session_caps_recording_length():
    session = _session(max_seconds=0.4)
    session.feed(_frame(7))
    assert session.feed(_loud(5)) == ["utterance"]


def test_session_reset_discards_recording():
    session = _session()
    session.feed(_frame(7))
    session.feed(_loud(2))
    session.reset()
    assert session.state == "listening"
    assert session.feed(_quiet(30)) == []


# --- /v1/speech/wake WebSocket ----------------------------------------------


def _wake_client(
    monkeypatch, *, transcript="what time is it", backend=True, vad=None, keyword=None
):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openjarvis.server.api_routes import speech_router
    from openjarvis.speech import wake_word

    monkeypatch.setattr(wake_word, "_load_model", lambda name: FakeModel())
    # Synthetic tones aren't speech to a real VAD; use the RMS fallback.
    monkeypatch.setattr(wake_word, "make_vad_speech_check", lambda: vad)
    monkeypatch.setattr(
        wake_word, "make_whisper_keyword_transcriber", lambda *a: keyword
    )
    app = FastAPI()
    app.include_router(speech_router)
    app.state.config = SimpleNamespace(
        speech=SimpleNamespace(wake_word="hey_jarvis", wake_threshold=0.5, language="")
    )
    transcribed: list[bytes] = []

    def transcribe(audio, **kwargs):
        transcribed.append(audio)
        return SimpleNamespace(text=transcript)

    app.state.speech_backend = (
        SimpleNamespace(backend_id="fake", transcribe=transcribe) if backend else None
    )
    return TestClient(app), transcribed


def test_wake_socket_full_flow(monkeypatch):
    client, transcribed = _wake_client(monkeypatch)
    with client.websocket_connect("/v1/speech/wake") as ws:
        assert ws.receive_json() == {"type": "ready", "wake_word": "hey_jarvis"}
        ws.send_bytes(_frame(7).tobytes())
        assert ws.receive_json() == {"type": "wake"}
        ws.send_bytes(_loud(3).tobytes())
        ws.send_bytes(_quiet(20).tobytes())
        assert ws.receive_json() == {"type": "transcribing"}
        assert ws.receive_json() == {"type": "transcript", "text": "what time is it"}
    assert len(transcribed) == 1 and transcribed[0][:4] == b"RIFF"


def test_wake_socket_reset_control(monkeypatch):
    client, transcribed = _wake_client(monkeypatch)
    with client.websocket_connect("/v1/speech/wake") as ws:
        ws.receive_json()
        ws.send_bytes(_frame(7).tobytes())
        assert ws.receive_json() == {"type": "wake"}
        ws.send_bytes(_loud(3).tobytes())
        ws.send_json({"type": "reset"})
        ws.send_bytes(_quiet(20).tobytes())
        ws.send_bytes(_frame(7).tobytes())
        assert ws.receive_json() == {"type": "wake"}
    assert transcribed == []


def test_wake_socket_rejects_bad_frames(monkeypatch):
    client, _ = _wake_client(monkeypatch)
    with client.websocket_connect("/v1/speech/wake") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x01\x02")
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(b"\x00" * (64 * 1024 + 2))
        assert ws.receive_json()["type"] == "error"


def test_wake_socket_without_speech_backend(monkeypatch):
    client, _ = _wake_client(monkeypatch, backend=False)
    with client.websocket_connect("/v1/speech/wake") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Speech backend" in msg["detail"]


# --- Noise after the wake word must not become a request -------------------


def test_session_single_noise_burst_is_not_an_utterance():
    """Regression: one loud noise frame after waking was sent as a request
    (and Whisper hallucinated text from it)."""
    session = _session(startup_silence_seconds=0.8, silence_seconds=0.24)
    session.feed(_frame(7))
    events = session.feed(np.concatenate([_quiet(2), _loud(1), _quiet(12)]))
    assert events == ["timeout"]
    assert session.take_wav() is None


def test_session_uses_injected_speech_check():
    """Speech is decided by the injected check (e.g. a VAD), not loudness."""
    from openjarvis.speech.wake_word import WakeSession

    is_speech = lambda frame: int(frame[1]) == 5  # noqa: E731
    session = WakeSession(
        WakeWordDetector(model=FakeModel()),
        is_speech=is_speech,
        silence_seconds=0.24,
    )
    session.feed(_frame(7))
    speech = _quiet(4)
    speech[1::FRAME_SAMPLES] = 5  # quiet but "speech" per the check
    # Loud frames the check rejects are noise: still recording, no event.
    assert session.feed(_loud(10)) == []
    assert session.state == "recording"
    assert session.feed(np.concatenate([speech, _quiet(3)])) == ["utterance"]


def test_wake_socket_uses_vad_speech_check(monkeypatch):
    """The server ends requests on VAD speech, so loud non-speech is ignored."""
    calls = []

    def vad(frame):
        calls.append(1)
        return False

    client, transcribed = _wake_client(monkeypatch, vad=vad)
    with client.websocket_connect("/v1/speech/wake") as ws:
        ws.receive_json()
        ws.send_bytes(_frame(7).tobytes())
        assert ws.receive_json() == {"type": "wake"}
        for _ in range(7):  # 5.6 s of loud audio the VAD rejects as non-speech
            ws.send_bytes(_loud(10).tobytes())
        assert ws.receive_json() == {"type": "timeout"}
    assert calls and transcribed == []


def test_vad_check_flags_real_speech_not_silence():
    pytest.importorskip("openwakeword")
    from openjarvis.speech.wake_word import make_vad_speech_check

    is_speech = make_vad_speech_check()
    assert is_speech is not None
    assert is_speech(_quiet(1)) is False


# --- Whisper fallback: accents the wake model misses ------------------------


def _speech_marked(n_frames: int) -> np.ndarray:
    """Frames the injected `_marker_speech` check treats as speech."""
    frames = _quiet(n_frames)
    frames[1::FRAME_SAMPLES] = 5
    return frames


def _marker_speech(frame) -> bool:
    return int(frame[1]) == 5


def _keyword_session(transcript, **kwargs):
    from openjarvis.speech.wake_word import WakeSession

    calls = []

    def transcribe(samples):
        calls.append(len(samples))
        return transcript

    session = WakeSession(
        WakeWordDetector(model=FakeModel()),
        is_speech=_marker_speech,
        keyword_transcriber=transcribe,
        **kwargs,
    )
    return session, calls


@pytest.mark.parametrize(
    "text",
    ["Hey, Jarvis.", "hey javis", "Hey Jarvis!", "Okay Jervis.", "Jarvis?"],
)
def test_wake_phrase_variants_match(text):
    from openjarvis.speech.wake_word import split_wake_phrase

    assert split_wake_phrase(text) == ""


@pytest.mark.parametrize(
    "text, rest",
    [
        ("Hey Jarvis, what's the weather?", "what's the weather?"),
        ("hey javis set a timer for five minutes", "set a timer for five minutes"),
    ],
)
def test_wake_phrase_returns_request(text, rest):
    from openjarvis.speech.wake_word import split_wake_phrase

    assert split_wake_phrase(text) == rest


@pytest.mark.parametrize(
    "text",
    ["", "you", "Hey there.", "Travis is here", "I told Jarvis already", "Jarvisa"],
)
def test_non_wake_phrases_rejected(text):
    from openjarvis.speech.wake_word import split_wake_phrase

    assert split_wake_phrase(text) is None


def test_keyword_fallback_wakes_on_missed_wake_word():
    session, calls = _keyword_session("Hey, Javis.")
    events = session.feed(np.concatenate([_quiet(3), _speech_marked(8), _quiet(6)]))
    assert events == ["wake"]
    assert session.state == "recording"
    assert len(calls) == 1


def test_keyword_fallback_one_breath_request():
    session, _ = _keyword_session("Hey Jarvis, what's the weather?")
    events = session.feed(np.concatenate([_speech_marked(20), _quiet(6)]))
    assert events == ["wake", "command"]
    assert session.take_text() == "what's the weather?"
    assert session.take_text() is None
    assert session.state == "listening"


def test_keyword_fallback_ignores_other_speech():
    session, calls = _keyword_session("Pass me the salt.")
    assert session.feed(np.concatenate([_speech_marked(10), _quiet(6)])) == []
    assert session.state == "listening"
    assert len(calls) == 1


def test_keyword_fallback_skips_long_speech():
    session, calls = _keyword_session("Hey Jarvis", max_keyword_seconds=1.0)
    session.feed(np.concatenate([_speech_marked(30), _quiet(6)]))
    assert calls == []


def test_keyword_fallback_skips_tiny_blips():
    session, calls = _keyword_session("Hey Jarvis")
    session.feed(np.concatenate([_speech_marked(1), _quiet(6)]))
    assert calls == []


def test_fast_path_does_not_call_transcriber():
    session, calls = _keyword_session("Hey Jarvis")
    assert session.feed(_frame(7)) == ["wake"]
    assert calls == []


def test_wake_socket_one_breath_command(monkeypatch):
    """'Hey Javis, what's the time' -> keyword fallback sends the request."""
    client, transcribed = _wake_client(
        monkeypatch,
        vad=_marker_speech,
        keyword=lambda samples: "Hey Javis, what's the time?",
    )
    with client.websocket_connect("/v1/speech/wake") as ws:
        ws.receive_json()
        ws.send_bytes(_speech_marked(15).tobytes())
        ws.send_bytes(_quiet(6).tobytes())
        assert ws.receive_json() == {"type": "wake"}
        assert ws.receive_json() == {"type": "transcript", "text": "what's the time?"}
    assert transcribed == []  # no second transcription needed


def test_repeated_wake_phrase_rewakes_instead_of_sending():
    """Saying 'Hey Jarvis' again after the chime must not become the request."""
    session, calls = _keyword_session("Hey Javis.", silence_seconds=0.24)
    assert session.feed(_frame(7)) == ["wake"]
    events = session.feed(np.concatenate([_speech_marked(10), _quiet(4)]))
    assert events == ["wake"]
    assert session.state == "recording"
    assert session.take_wav() is None
    assert len(calls) == 1


def test_short_request_still_sent_after_wake_check():
    session, _ = _keyword_session("What's the time?", silence_seconds=0.24)
    session.feed(_frame(7))
    events = session.feed(np.concatenate([_speech_marked(10), _quiet(4)]))
    assert events == ["utterance"]
    assert session.take_wav() is not None


def test_long_request_skips_wake_check():
    session, calls = _keyword_session(
        "Hey Jarvis", silence_seconds=0.24, rewake_max_seconds=1.0
    )
    session.feed(_frame(7))
    assert session.feed(np.concatenate([_speech_marked(20), _quiet(4)])) == [
        "utterance"
    ]
    assert calls == []


def test_socket_strips_leading_wake_phrase_from_request(monkeypatch):
    client, _ = _wake_client(monkeypatch, transcript="Hey Jarvis, what time is it")
    with client.websocket_connect("/v1/speech/wake") as ws:
        ws.receive_json()
        ws.send_bytes(_frame(7).tobytes())
        assert ws.receive_json() == {"type": "wake"}
        ws.send_bytes(_loud(3).tobytes())
        ws.send_bytes(_quiet(20).tobytes())
        assert ws.receive_json() == {"type": "transcribing"}
        assert ws.receive_json() == {"type": "transcript", "text": "what time is it"}


def test_rewake_after_long_pause_before_repeat():
    """Regression (real capture): pause, then 'Hey Jarvis' again -> re-wake."""
    session, _ = _keyword_session(
        "Hey Javis.", silence_seconds=0.24, rewake_max_seconds=1.0
    )
    session.feed(_frame(7))
    events = session.feed(np.concatenate([_quiet(30), _speech_marked(8), _quiet(4)]))
    assert events == ["wake"]
