from pathlib import Path

from app.audio_pipeline import AudioProcessingOptions, run_single_audio_input
from app.models import TranscriptSegment


class FakeAudioInfo:
    def __init__(self, duration_sec: float) -> None:
        self.duration_sec = duration_sec


class FakeAudioTranscriber:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe_video(self, video_path: Path):
        self.calls.append(Path(video_path))
        return [
            TranscriptSegment(start_sec=0.0, end_sec=1.5, text="こんにちは"),
            TranscriptSegment(start_sec=1.5, end_sec=3.0, text="テストです"),
        ], FakeAudioInfo(duration_sec=3.0)


def test_run_single_audio_input_writes_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    output_dir = tmp_path / "output"
    download_dir = tmp_path / "downloads"
    runtime_dir = tmp_path / "runtime"
    transcriber = FakeAudioTranscriber()

    result, paths = run_single_audio_input(
        input_ref=str(source),
        audio_transcriber=transcriber,
        options=AudioProcessingOptions(runtime_video_dir=runtime_dir),
        output_dir=output_dir,
        download_dir=download_dir,
    )

    assert result.transcription_mode == "asr"
    assert result.duration_sec == 3.0
    assert [segment.text for segment in result.segments] == ["こんにちは", "テストです"]
    assert len(transcriber.calls) == 1
    assert paths["json"].exists()
    assert paths["txt"].exists()
    assert paths["srt"].exists()
