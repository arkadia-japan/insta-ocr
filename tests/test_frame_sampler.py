import cv2
import numpy as np

from app.frame_sampler import (
    MAX_CANDIDATE_FRAMES,
    VisualSegment,
    _frame_quality_score,
    _prepare_frame_for_storage,
    _should_start_new_segment,
    merge_adjacent_visual_segments,
)


def test_should_start_new_segment_when_visual_change_is_large():
    assert _should_start_new_segment(
        segment_start_sec=0.0,
        current_time_sec=1.0,
        scene_delta=0.18,
        visual_delta=0.16,
        pixel_delta=0.05,
        scene_threshold=0.22,
        text_change_threshold=0.055,
        min_interval_sec=0.4,
    ) is True


def test_should_not_start_new_segment_for_small_same_slide_change():
    assert _should_start_new_segment(
        segment_start_sec=0.0,
        current_time_sec=1.0,
        scene_delta=0.19,
        visual_delta=0.03,
        pixel_delta=0.002,
        scene_threshold=0.22,
        text_change_threshold=0.055,
        min_interval_sec=0.4,
    ) is False


def test_merge_adjacent_visual_segments_collapses_nearly_identical_frames():
    base_frame = np.full((40, 40, 3), 120, dtype=np.uint8)
    similar_frame = np.full((40, 40, 3), 121, dtype=np.uint8)
    different_frame = np.full((40, 40, 3), 10, dtype=np.uint8)

    segments = [
        VisualSegment(0.0, 1.0, 0.2, 0.1, 0.04, 10.0, base_frame, [base_frame]),
        VisualSegment(1.0, 2.0, 1.2, 0.1, 0.04, 12.0, similar_frame, [similar_frame]),
        VisualSegment(2.0, 3.0, 2.2, 0.4, 0.2, 8.0, different_frame, [different_frame]),
    ]

    merged = merge_adjacent_visual_segments(segments, visual_threshold=0.1, pixel_threshold=0.02)

    assert len(merged) == 2
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 2.0


def test_frame_quality_score_prefers_clear_text_frame():
    clear_frame = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.putText(clear_frame, "TEXT", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
    blurred_frame = cv2.GaussianBlur(clear_frame, (9, 9), 0)

    assert _frame_quality_score(clear_frame) > _frame_quality_score(blurred_frame)


def test_prepare_frame_for_storage_downscales_large_frames():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    prepared = _prepare_frame_for_storage(frame)

    assert max(prepared.shape[:2]) == 1280


def test_merge_adjacent_visual_segments_limits_candidate_count():
    base_frame = np.full((80, 80, 3), 120, dtype=np.uint8)
    other_frames = [np.full((80, 80, 3), value, dtype=np.uint8) for value in (121, 122, 123, 124)]
    segments = [
        VisualSegment(0.0, 1.0, 0.2, 0.1, 0.04, 10.0, base_frame, [base_frame, other_frames[0]]),
        VisualSegment(1.0, 2.0, 1.2, 0.1, 0.04, 12.0, other_frames[1], [other_frames[1], other_frames[2]]),
        VisualSegment(2.0, 3.0, 2.2, 0.4, 0.2, 8.0, other_frames[3], [other_frames[3]]),
    ]

    merged = merge_adjacent_visual_segments(segments, visual_threshold=0.1, pixel_threshold=0.02)

    assert len(merged[0].candidate_frames) == MAX_CANDIDATE_FRAMES
