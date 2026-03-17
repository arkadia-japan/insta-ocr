from pathlib import Path

import pytest

from app.audio_transcriber import AudioTranscriber


class _UnusedModel:
    def transcribe(self, *_args, **_kwargs):
        raise AssertionError("model.transcribe should not be called")


class _FailingModel:
    def transcribe(self, *_args, **_kwargs):
        raise ValueError("backend boom")


def _build_transcriber(model) -> AudioTranscriber:
    transcriber = AudioTranscriber.__new__(AudioTranscriber)
    transcriber.model = model
    transcriber.language = None
    return transcriber


def test_transcribe_video_rejects_inputs_without_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    transcriber = _build_transcriber(_UnusedModel())
    monkeypatch.setattr("app.audio_transcriber._has_audio_stream", lambda _path: False)

    with pytest.raises(RuntimeError, match="no audio stream"):
        transcriber.transcribe_video(Path("video.mp4"))


def test_transcribe_video_wraps_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    transcriber = _build_transcriber(_FailingModel())
    monkeypatch.setattr("app.audio_transcriber._has_audio_stream", lambda _path: True)

    with pytest.raises(RuntimeError, match="backend boom"):
        transcriber.transcribe_video(Path("video.mp4"))
