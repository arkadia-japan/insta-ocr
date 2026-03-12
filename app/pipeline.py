from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .downloader import download_video
from .exporters import write_outputs
from .frame_sampler import merge_adjacent_visual_segments, sample_video_segments
from .models import TranscriptSegment, TranscriptionResult
from .ocr_engine import OcrEngine
from .postprocess import consolidate_ocr_candidates, harmonize_segment_lines, merge_adjacent_similar_segments
from .text_corrections import apply_text_corrections, load_text_corrections
from .utils import detect_platform, ensure_directory, is_url, safe_stem_from_input

StatusCallback = Callable[[str], None]


@dataclass
class ProcessingOptions:
    sample_fps: float = 4.0
    scene_threshold: float = 0.22
    text_change_threshold: float = 0.055
    min_interval_sec: float = 0.4
    similarity_threshold: float = 0.86
    json_indent: int = 2
    keep_video: bool = False
    retry_on_empty: bool = True
    cookies_file: Path | None = None
    corrections_file: Path | None = None


def resolve_video_path(
    input_ref: str,
    download_dir: Path,
    cookies_file: Path | None,
) -> tuple[Path, bool]:
    if is_url(input_ref):
        video_path = download_video(input_ref, download_dir=download_dir, cookies_file=cookies_file)
        return video_path, True

    local_path = Path(input_ref).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Input file was not found: {local_path}")
    return local_path, False


def run_single_input(
    input_ref: str,
    ocr_engine: OcrEngine,
    options: ProcessingOptions,
    output_dir: Path,
    download_dir: Path,
    status_callback: StatusCallback | None = None,
) -> tuple[TranscriptionResult, dict[str, Path]]:
    ensure_directory(output_dir)
    ensure_directory(download_dir)

    corrections = load_text_corrections(correction_file=options.corrections_file)

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

    _report_status(status_callback, "画像の切り替えを解析しています")
    visual_segments, duration_sec = sample_video_segments(
        video_path=video_path,
        sample_fps=options.sample_fps,
        scene_threshold=options.scene_threshold,
        text_change_threshold=options.text_change_threshold,
        min_interval_sec=options.min_interval_sec,
    )
    visual_segments = merge_adjacent_visual_segments(visual_segments)

    _report_status(status_callback, f"OCRを実行しています: {len(visual_segments)} セグメント")
    segments = _extract_segments(
        ocr_engine=ocr_engine,
        visual_segments=visual_segments,
        status_callback=status_callback,
    )

    fallback_used = False
    sampled_frames = len(visual_segments)
    ocr_hits = len(segments)
    if not segments and options.retry_on_empty:
        fallback_used = True
        _report_status(status_callback, "結果が空だったため、より細かい設定で再試行しています")
        retry_visual_segments, _ = sample_video_segments(
            video_path=video_path,
            sample_fps=max(options.sample_fps, 5.0),
            scene_threshold=min(options.scene_threshold, 0.16),
            text_change_threshold=min(options.text_change_threshold, 0.035),
            min_interval_sec=min(options.min_interval_sec, 0.2),
        )
        retry_visual_segments = merge_adjacent_visual_segments(retry_visual_segments)
        retry_segments = _extract_segments(
            ocr_engine=ocr_engine,
            visual_segments=retry_visual_segments,
            status_callback=status_callback,
        )
        if retry_segments:
            segments = retry_segments
            sampled_frames = len(retry_visual_segments)
            ocr_hits = len(retry_segments)

    corrected_segments = [
        TranscriptSegment(
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            text=apply_text_corrections(segment.text, corrections),
            confidence=segment.confidence,
        )
        for segment in segments
    ]
    merged_segments = merge_adjacent_similar_segments(corrected_segments)
    harmonized_segments = harmonize_segment_lines(merged_segments)

    result = TranscriptionResult(
        input_ref=input_ref,
        platform=platform,
        video_path=str(video_path),
        duration_sec=duration_sec,
        sampled_frames=sampled_frames,
        ocr_hits=ocr_hits,
        fallback_used=fallback_used,
        segments=harmonized_segments,
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

    _report_status(status_callback, "完了しました")
    return result, paths


def _extract_segments(
    ocr_engine: OcrEngine,
    visual_segments,
    status_callback: StatusCallback | None = None,
) -> list[TranscriptSegment]:
    if not visual_segments:
        return []

    frames_to_ocr = []
    frame_spans: list[tuple[int, int]] = []
    for segment in visual_segments:
        candidate_frames = segment.candidate_frames or [segment.frame]
        start = len(frames_to_ocr)
        frames_to_ocr.extend(candidate_frames)
        frame_spans.append((start, len(candidate_frames)))

    _report_status(
        status_callback,
        f"OCR中: {len(visual_segments)} セグメント / {len(frames_to_ocr)} フレーム",
    )
    batch_results = ocr_engine.extract_text_batch(frames_to_ocr)

    segments: list[TranscriptSegment] = []
    for visual_segment, (start, length) in zip(visual_segments, frame_spans):
        text, confidence = consolidate_ocr_candidates(batch_results[start : start + length])
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_sec=visual_segment.start_sec,
                end_sec=max(visual_segment.end_sec, visual_segment.start_sec + 0.1),
                text=text,
                confidence=confidence,
            )
        )
    return segments


def _report_status(callback: StatusCallback | None, message: str) -> None:
    if callback:
        callback(message)
