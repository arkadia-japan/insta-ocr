from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .downloader import download_video
from .exporters import write_outputs
from .frame_sampler import merge_adjacent_visual_segments, sample_video_segments
from .models import TranscriptSegment, TranscriptionResult
from .ocr_engine import OcrEngine
from .postprocess import consolidate_ocr_candidates, harmonize_segment_lines, merge_adjacent_similar_segments
from .runtime_paths import stage_video_for_runtime
from .text_corrections import apply_text_corrections, load_text_corrections
from .utils import detect_platform, ensure_directory, is_url, looks_like_image_url, normalize_text, safe_stem_from_input

StatusCallback = Callable[[str], None]
FrameSignature = tuple[int, int, bytes]


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
    runtime_video_dir: Path | None = None


def resolve_video_path(
    input_ref: str,
    download_dir: Path,
    cookies_file: Path | None,
) -> tuple[Path, bool]:
    if is_url(input_ref):
        if looks_like_image_url(input_ref):
            raise RuntimeError(
                "Input URL looks like an image thumbnail, not a video URL. "
                "Check the sheet URL column or source data."
            )
        video_path = download_video(input_ref, download_dir=download_dir, cookies_file=cookies_file)
        return video_path, True

    local_path = Path(input_ref).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"??????????????: {local_path}")
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
        _report_status(status_callback, f"??URL????????: {platform}")
    else:
        _report_status(status_callback, "???????????????")

    video_path, was_downloaded = resolve_video_path(
        input_ref=input_ref,
        download_dir=download_dir,
        cookies_file=options.cookies_file,
    )
    processing_video_path = stage_video_for_runtime(
        video_path=video_path,
        runtime_video_dir=options.runtime_video_dir or download_dir,
    )

    _report_status(status_callback, "????????????????")
    visual_segments, duration_sec = sample_video_segments(
        video_path=processing_video_path,
        sample_fps=options.sample_fps,
        scene_threshold=options.scene_threshold,
        text_change_threshold=options.text_change_threshold,
        min_interval_sec=options.min_interval_sec,
    )
    visual_segments = merge_adjacent_visual_segments(visual_segments)

    segments = _extract_segments(
        ocr_engine=ocr_engine,
        visual_segments=visual_segments,
        status_callback=status_callback,
    )

    fallback_used = False
    sampled_frames = len(visual_segments)
    ocr_hits = len(segments)
    retry_reason = ""
    if not segments:
        retry_reason = "??????????????????????????"
    elif _should_retry_weak_output(segments):
        retry_reason = "OCR??????????????????????????????????"

    if retry_reason and options.retry_on_empty:
        fallback_used = True
        _report_status(status_callback, retry_reason)
        retry_visual_segments, _ = sample_video_segments(
            video_path=processing_video_path,
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
        if _score_ocr_segments(retry_segments) > _score_ocr_segments(segments):
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
    merged_segments = merge_adjacent_similar_segments(
        corrected_segments,
        similarity_threshold=options.similarity_threshold,
    )
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
        transcription_mode="ocr",
    )

    stem = safe_stem_from_input(input_ref)
    _report_status(status_callback, "??????????????")
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

    _report_status(status_callback, "??????")
    return result, paths


def _extract_segments(
    ocr_engine: OcrEngine,
    visual_segments,
    status_callback: StatusCallback | None = None,
) -> list[TranscriptSegment]:
    if not visual_segments:
        return []

    ocr_cache: dict[FrameSignature, tuple[str, float | None]] = {}
    representative_frames = [segment.frame for segment in visual_segments]
    _report_status(
        status_callback,
        f"OCR?: {len(visual_segments)} ????? / {len(representative_frames)} ??????",
    )
    primary_results = _extract_results_for_frames(
        ocr_engine=ocr_engine,
        frames=representative_frames,
        memo=ocr_cache,
        status_callback=status_callback,
        progress_label="primary",
    )

    rescue_plan: list[tuple[int, list[object]]] = []
    rescue_frames: list[object] = []
    for segment_index, (segment, primary_result) in enumerate(zip(visual_segments, primary_results)):
        if not _should_rescue_segment(primary_result):
            continue
        additional_frames = _select_additional_candidate_frames(segment)
        if not additional_frames:
            continue
        rescue_plan.append((segment_index, additional_frames))
        rescue_frames.extend(additional_frames)

    rescue_results: list[tuple[str, float | None]] = []
    if rescue_frames:
        _report_status(
            status_callback,
            f"OCR???: {len(rescue_plan)} ????? / {len(rescue_frames)} ??????",
        )
        rescue_results = _extract_results_for_frames(
            ocr_engine=ocr_engine,
            frames=rescue_frames,
            memo=ocr_cache,
            status_callback=status_callback,
            progress_label="rescue",
        )

    rescue_results_by_segment: dict[int, list[tuple[str, float | None]]] = {}
    rescue_offset = 0
    for segment_index, additional_frames in rescue_plan:
        next_offset = rescue_offset + len(additional_frames)
        rescue_results_by_segment[segment_index] = rescue_results[rescue_offset:next_offset]
        rescue_offset = next_offset

    segments: list[TranscriptSegment] = []
    for segment_index, visual_segment in enumerate(visual_segments):
        candidate_results = [primary_results[segment_index]]
        candidate_results.extend(rescue_results_by_segment.get(segment_index, []))
        text, confidence = consolidate_ocr_candidates(candidate_results)
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


def _extract_results_for_frames(
    ocr_engine: OcrEngine,
    frames: list[object],
    memo: dict[FrameSignature, tuple[str, float | None]],
    fast_only: bool = False,
    status_callback: StatusCallback | None = None,
    progress_label: str = "",
) -> list[tuple[str, float | None]]:
    if not frames:
        return []

    results: list[tuple[str, float | None] | None] = [None] * len(frames)
    unique_frames: list[object] = []
    unique_keys: list[FrameSignature] = []
    unique_slots: list[list[int]] = []
    pending_by_key: dict[FrameSignature, int] = {}

    for index, frame in enumerate(frames):
        key = _frame_signature_key(frame)
        cached = memo.get(key)
        if cached is not None:
            results[index] = cached
            continue

        pending_index = pending_by_key.get(key)
        if pending_index is not None:
            unique_slots[pending_index].append(index)
            continue

        pending_by_key[key] = len(unique_frames)
        unique_frames.append(frame)
        unique_keys.append(key)
        unique_slots.append([index])

    if unique_frames:
        batch_size = _recommended_ocr_batch_size(ocr_engine=ocr_engine, total_frames=len(unique_frames))
        for start in range(0, len(unique_frames), batch_size):
            end = min(start + batch_size, len(unique_frames))
            frame_chunk = unique_frames[start:end]
            key_chunk = unique_keys[start:end]
            slot_chunk = unique_slots[start:end]

            if fast_only and hasattr(ocr_engine, "extract_text_primary_batch"):
                extracted = ocr_engine.extract_text_primary_batch(frame_chunk)
            else:
                extracted = ocr_engine.extract_text_batch(frame_chunk)

            for key, slots, result in zip(key_chunk, slot_chunk, extracted):
                memo[key] = result
                for index in slots:
                    results[index] = result

            if status_callback and len(unique_frames) > batch_size:
                prefix = f"OCR {progress_label}".strip()
                _report_status(status_callback, f"{prefix}: {end}/{len(unique_frames)} frames")

    return [result if result is not None else ("", None) for result in results]


def _recommended_ocr_batch_size(ocr_engine: OcrEngine, total_frames: int) -> int:
    backend_name = getattr(ocr_engine, "_backend_name", "")
    if backend_name == "easyocr":
        return 4
    return max(total_frames, 1)


def _should_rescue_segment(result: tuple[str, float | None]) -> bool:
    text, confidence = result
    normalized = normalize_text(text)
    if not normalized:
        return True

    line_count = len([line for line in text.splitlines() if normalize_text(line)])
    char_count = len(normalized.replace(" ", ""))
    noise_count = _text_noise_count(text)
    return (
        confidence is None
        or confidence < 0.74
        or line_count <= 1
        or (line_count <= 2 and char_count < 18)
        or noise_count >= 2
    )


def _select_additional_candidate_frames(segment, max_frames: int = 2) -> list[object]:
    primary_key = _frame_signature_key(segment.frame)
    seen_keys: set[FrameSignature] = {primary_key}
    scored_frames: list[tuple[float, object]] = []

    for frame in segment.candidate_frames:
        key = _frame_signature_key(frame)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        scored_frames.append((_frame_difference_score(segment.frame, frame), frame))

    scored_frames.sort(key=lambda item: item[0], reverse=True)
    return [frame for _, frame in scored_frames[:max_frames]]


def _frame_signature_key(frame) -> FrameSignature:
    small = _small_grayscale(frame, size=40)
    height, width = frame.shape[:2]
    return height, width, small.tobytes()


def _frame_difference_score(left_frame, right_frame) -> float:
    left_small = _small_grayscale(left_frame, size=40).astype(np.float32) / 255.0
    right_small = _small_grayscale(right_frame, size=40).astype(np.float32) / 255.0
    return float(np.mean(np.abs(left_small - right_small)))


def _small_grayscale(frame, size: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def _text_noise_count(text: str) -> int:
    return sum(char in "|[]{}<>~`" for char in text) + sum(
        char.isascii() and not (char.isalnum() or char.isspace() or char in "!?.,:;/-_()#%&\'\"")
        for char in text
    )


def _should_retry_weak_output(segments: list[TranscriptSegment]) -> bool:
    non_empty_segments = [segment for segment in segments if normalize_text(segment.text)]
    if not non_empty_segments:
        return True

    combined_text = "\n".join(segment.text for segment in non_empty_segments)
    normalized = normalize_text(combined_text)
    char_count = len(normalized.replace(" ", ""))
    line_count = len(non_empty_segments)
    confidences = [float(segment.confidence) for segment in non_empty_segments if segment.confidence is not None]
    average_confidence = (sum(confidences) / len(confidences)) if confidences else None
    noise_count = _text_noise_count(combined_text)

    return (
        char_count < 4
        or (line_count == 1 and char_count < 10)
        or (average_confidence is not None and average_confidence < 0.58)
        or (average_confidence is not None and char_count < 12 and average_confidence < 0.74)
        or noise_count >= 3
    )


def _score_ocr_segments(segments: list[TranscriptSegment]) -> float:
    non_empty_segments = [segment for segment in segments if normalize_text(segment.text)]
    if not non_empty_segments:
        return 0.0

    combined_text = "\n".join(segment.text for segment in non_empty_segments)
    normalized = normalize_text(combined_text)
    char_count = len(normalized.replace(" ", ""))
    line_count = len(non_empty_segments)
    confidences = [float(segment.confidence) for segment in non_empty_segments if segment.confidence is not None]
    average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.55
    noise_penalty = _text_noise_count(combined_text) * 12
    return average_confidence * 100.0 + min(char_count, 140) + line_count * 8 - noise_penalty


def _report_status(callback: StatusCallback | None, message: str) -> None:
    if callback:
        callback(message)
