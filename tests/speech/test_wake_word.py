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


def _wake_client(monkeypatch, *, transcript="what time is it", backend=True):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openjarvis.server.api_routes import speech_router
    from openjarvis.speech import wake_word

    monkeypatch.setattr(wake_word, "_load_model", lambda name: FakeModel())
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
