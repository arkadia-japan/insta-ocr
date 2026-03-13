from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VisualSegment:
    start_sec: float
    end_sec: float
    representative_time_sec: float
    scene_delta: float
    visual_delta: float
    quality_score: float
    frame: np.ndarray
    candidate_frames: list[np.ndarray]


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    normalized = cv2.normalize(hist, hist).flatten()
    return normalized


def _visual_signature(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    normalized = small.astype(np.float32) / 255.0
    diff_hash = (small[:, 1:] > small[:, :-1]).flatten()
    return normalized, diff_hash


def _frame_quality_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    y0 = int(height * 0.05)
    y1 = max(y0 + 1, int(height * 0.95))
    x0 = int(width * 0.05)
    x1 = max(x0 + 1, int(width * 0.95))
    focus = gray[y0:y1, x0:x1]
    if focus.size == 0:
        focus = gray

    laplacian_var = float(cv2.Laplacian(focus, cv2.CV_64F).var())
    contrast = float(np.std(focus))
    edges = cv2.Canny(focus, 70, 160)
    edge_density = float(np.mean(edges) / 255.0)
    mean_luma = float(np.mean(focus) / 255.0)
    exposure_balance = max(0.0, 1.0 - abs(mean_luma - 0.62))
    return (laplacian_var * 0.75) + (contrast * 3.0) + (edge_density * 180.0) + (exposure_balance * 24.0)


def sample_video_segments(
    video_path: Path,
    sample_fps: float = 4.0,
    scene_threshold: float = 0.22,
    text_change_threshold: float = 0.055,
    min_interval_sec: float = 0.4,
) -> tuple[list[VisualSegment], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_frames / fps) if total_frames > 0 else 0.0
    frame_step = max(1, int(round(fps / max(sample_fps, 0.1))))

    segments: list[VisualSegment] = []
    current_segment: dict | None = None
    previous_hist: np.ndarray | None = None
    previous_small: np.ndarray | None = None
    previous_hash: np.ndarray | None = None
    frame_index = -1

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_index += 1
        if frame_index % frame_step != 0:
            continue

        time_sec = frame_index / fps
        current_hist = _frame_histogram(frame)
        current_small, current_hash = _visual_signature(frame)
        quality_score = _frame_quality_score(frame)

        if previous_hist is None or previous_small is None or previous_hash is None:
            scene_delta = 1.0
            visual_delta = 1.0
            pixel_delta = 1.0
        else:
            scene_delta = float(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA))
            visual_delta = float(np.mean(previous_hash != current_hash))
            pixel_delta = float(np.mean(np.abs(previous_small - current_small)))

        if current_segment is None:
            current_segment = _start_segment(
                time_sec=time_sec,
                scene_delta=scene_delta,
                visual_delta=visual_delta,
                quality_score=quality_score,
                frame=frame,
            )
        elif _should_start_new_segment(
            segment_start_sec=current_segment["start_sec"],
            current_time_sec=time_sec,
            scene_delta=scene_delta,
            visual_delta=visual_delta,
            pixel_delta=pixel_delta,
            scene_threshold=scene_threshold,
            text_change_threshold=text_change_threshold,
            min_interval_sec=min_interval_sec,
        ):
            segments.append(_finalize_segment(current_segment, time_sec))
            current_segment = _start_segment(
                time_sec=time_sec,
                scene_delta=scene_delta,
                visual_delta=visual_delta,
                quality_score=quality_score,
                frame=frame,
            )
        else:
            current_segment["scene_delta"] = max(current_segment["scene_delta"], scene_delta)
            current_segment["visual_delta"] = max(current_segment["visual_delta"], visual_delta)
            if quality_score > current_segment["quality_score"]:
                current_segment["representative_time_sec"] = time_sec
                current_segment["quality_score"] = quality_score
                current_segment["frame"] = frame.copy()
            _update_candidate_frames(
                current_segment=current_segment,
                frame=frame,
                time_sec=time_sec,
                quality_score=quality_score,
            )

        previous_hist = current_hist
        previous_small = current_small
        previous_hash = current_hash

    cap.release()

    if duration_sec <= 0 and frame_index >= 0:
        duration_sec = frame_index / fps

    if current_segment is not None:
        segments.append(_finalize_segment(current_segment, duration_sec))

    return segments, duration_sec


def merge_adjacent_visual_segments(
    segments: list[VisualSegment],
    visual_threshold: float = 0.1,
    pixel_threshold: float = 0.015,
) -> list[VisualSegment]:
    if not segments:
        return []

    merged_segments: list[VisualSegment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged_segments[-1]
        previous_small, previous_hash = _visual_signature(previous.frame)
        current_small, current_hash = _visual_signature(segment.frame)
        visual_delta = float(np.mean(previous_hash != current_hash))
        pixel_delta = float(np.mean(np.abs(previous_small - current_small)))

        if visual_delta <= visual_threshold and pixel_delta <= pixel_threshold:
            representative = previous if previous.quality_score >= segment.quality_score else segment
            merged_segments[-1] = VisualSegment(
                start_sec=previous.start_sec,
                end_sec=segment.end_sec,
                representative_time_sec=representative.representative_time_sec,
                scene_delta=max(previous.scene_delta, segment.scene_delta),
                visual_delta=max(previous.visual_delta, segment.visual_delta),
                quality_score=max(previous.quality_score, segment.quality_score),
                frame=representative.frame,
                candidate_frames=_merge_candidate_frame_lists(previous.candidate_frames, segment.candidate_frames),
            )
            continue

        merged_segments.append(segment)

    return merged_segments


def _start_segment(
    time_sec: float,
    scene_delta: float,
    visual_delta: float,
    quality_score: float,
    frame,
) -> dict:
    segment = {
        "start_sec": time_sec,
        "representative_time_sec": time_sec,
        "scene_delta": scene_delta,
        "visual_delta": visual_delta,
        "quality_score": quality_score,
        "frame": frame.copy(),
        "candidate_frames": [],
    }
    _update_candidate_frames(
        current_segment=segment,
        frame=frame,
        time_sec=time_sec,
        quality_score=quality_score,
    )
    return segment


def _update_candidate_frames(
    current_segment: dict,
    frame,
    time_sec: float,
    quality_score: float,
    max_candidates: int = 3,
    min_time_gap_sec: float = 0.45,
) -> None:
    for candidate in current_segment["candidate_frames"]:
        if abs(candidate["time_sec"] - time_sec) < min_time_gap_sec:
            if quality_score > candidate["quality_score"]:
                candidate["time_sec"] = time_sec
                candidate["quality_score"] = quality_score
                candidate["frame"] = frame.copy()
            return

    current_segment["candidate_frames"].append(
        {
            "time_sec": time_sec,
            "quality_score": quality_score,
            "frame": frame.copy(),
        }
    )
    current_segment["candidate_frames"].sort(key=lambda item: item["quality_score"], reverse=True)
    del current_segment["candidate_frames"][max_candidates:]


def _finalize_segment(segment: dict, end_sec: float) -> VisualSegment:
    candidate_frames = [
        candidate["frame"]
        for candidate in sorted(segment["candidate_frames"], key=lambda item: item["time_sec"])
    ]
    if not candidate_frames:
        candidate_frames = [segment["frame"]]

    return VisualSegment(
        start_sec=segment["start_sec"],
        end_sec=max(segment["start_sec"], end_sec),
        representative_time_sec=segment["representative_time_sec"],
        scene_delta=segment["scene_delta"],
        visual_delta=segment["visual_delta"],
        quality_score=segment["quality_score"],
        frame=segment["frame"],
        candidate_frames=candidate_frames,
    )


def _should_start_new_segment(
    segment_start_sec: float,
    current_time_sec: float,
    scene_delta: float,
    visual_delta: float,
    pixel_delta: float,
    scene_threshold: float,
    text_change_threshold: float,
    min_interval_sec: float,
) -> bool:
    if current_time_sec - segment_start_sec < min_interval_sec:
        return False

    strong_scene = scene_delta >= max(scene_threshold * 2.0, scene_threshold + 0.18)
    strong_visual = visual_delta >= max(text_change_threshold * 1.9, text_change_threshold + 0.04)
    strong_pixel = pixel_delta >= 0.03
    combo_change = visual_delta >= text_change_threshold and scene_delta >= (scene_threshold * 0.75)
    return strong_scene or strong_visual or strong_pixel or combo_change


def _merge_candidate_frame_lists(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    max_candidates: int = 3,
) -> list[np.ndarray]:
    merged = [frame.copy() for frame in left_frames]
    merged.extend(frame.copy() for frame in right_frames)
    if len(merged) <= max_candidates:
        return merged

    scored = sorted(
        ((frame, _frame_quality_score(frame)) for frame in merged),
        key=lambda item: item[1],
        reverse=True,
    )
    return [frame for frame, _ in scored[:max_candidates]]
