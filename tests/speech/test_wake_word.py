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
