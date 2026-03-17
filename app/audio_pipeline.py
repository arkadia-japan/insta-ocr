from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio_transcriber import AudioTranscriber
from .exporters import write_outputs
from .models import TranscriptionResult
from .pipeline import resolve_video_path
from .runtime_paths import stage_video_for_runtime
from .utils import detect_platform, ensure_directory, is_url, safe_stem_from_input

StatusCallback = Callable[[str], None]


@dataclass
class AudioProcessingOptions:
    keep_video: bool = False
    cookies_file: Path | None = None
    runtime_video_dir: Path | None = None
    json_indent: int = 2


def run_single_audio_input(
    input_ref: str,
    audio_transcriber: AudioTranscriber,
    options: AudioProcessingOptions,
    output_dir: Path,
    download_dir: Path,
    status_callback: StatusCallback | None = None,
) -> tuple[TranscriptionResult, dict[str, Path]]:
    ensure_directory(output_dir)
    ensure_directory(download_dir)

    platform = detect_platform(input_ref)
    if is_url(input_ref):
        _report_status(status_callback, f"動画URLを取得しています: {platform}")
    else:
        _report_status(status_callback, "ローカル動画を読み込んでいます")

    video_path, was_downloaded = resolve_video_path(
        input_ref=input_ref,
        download_dir=download_dir,
        cookies_file=options.cookies_file,
    )
    processing_video_path = stage_video_for_runtime(
        video_path=video_path,
        runtime_video_dir=options.runtime_video_dir or download_dir,
    )

    _report_status(status_callback, "音声認識を実行しています")
    segments, info = audio_transcriber.transcribe_video(processing_video_path)
    duration_sec = info.duration_sec

    result = TranscriptionResult(
        input_ref=input_ref,
        platform=platform,
        video_path=str(video_path),
        duration_sec=duration_sec,
        sampled_frames=0,
        ocr_hits=len(segments),
        fallback_used=False,
        segments=segments,
        transcription_mode="asr",
    )

    stem = safe_stem_from_input(input_ref)
    _report_status(status_callback, "出力ファイルを書き出しています")
    paths = write_outputs(
        result=result,
        output_dir=output_dir,
        stem=stem,
        json_indent=options.json_indent,
    )

    if was_downloaded and not options.keep_video:
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass
    if processing_video_path != video_path:
        try:
            processing_video_path.unlink(missing_ok=True)
        except OSError:
            pass

    _report_status(status_callback, "完了しました")
    return result, paths


def _report_status(callback: StatusCallback | None, message: str) -> None:
    if callback:
        callback(message)
