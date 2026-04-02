import numpy as np
import pytest

from app.frame_sampler import VisualSegment
from app.models import TranscriptSegment
from app.pipeline import (
    _extract_results_for_frames,
    _extract_segments,
    _score_ocr_segments,
    _should_retry_weak_output,
    resolve_video_path,
)


class FakeOcrEngine:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls: list[tuple[str, list[int]]] = []

    def extract_text_batch(self, frames):
        frame_ids = [int(frame[0, 0, 0]) for frame in frames]
        self.calls.append(("full", frame_ids))
        return [self.mapping[frame_id] for frame_id in frame_ids]



def _frame(frame_id: int) -> np.ndarray:
    return np.full((8, 8, 3), frame_id, dtype=np.uint8)


def test_extract_segments_uses_representative_frames_first_and_reuses_cached_rescue_results():
    engine = FakeOcrEngine(
        {
            10: ("Strong title\nLine A\nLine B", 0.93),
            20: ("Slide title", 0.58),
            21: ("Slide title\nLine A\nLine B\nCall to action", 0.88),
        }
    )
    segments = [
        VisualSegment(0.0, 1.0, 0.2, 0.12, 0.05, 10.0, _frame(10), [_frame(10), _frame(11)]),
        VisualSegment(1.0, 2.0, 1.2, 0.18, 0.08, 9.0, _frame(20), [_frame(20), _frame(21)]),
        VisualSegment(2.0, 3.0, 2.2, 0.14, 0.04, 9.5, _frame(21), [_frame(21)]),
    ]

    transcript_segments = _extract_segments(engine, segments)

    assert engine.calls == [("full", [10, 20, 21])]
    assert [segment.text for segment in transcript_segments] == [
        "Strong title\nLine A\nLine B",
        "Slide title\nLine A\nLine B\nCall to action",
        "Slide title\nLine A\nLine B\nCall to action",
    ]


def test_extract_results_for_frames_uses_full_batch_when_fast_batch_is_unavailable():
    engine = FakeOcrEngine({30: ("Supplemental text", 0.8)})

    results = _extract_results_for_frames(
        ocr_engine=engine,
        frames=[_frame(30)],
        memo={},
        fast_only=True,
    )

    assert results == [("Supplemental text", 0.8)]
    assert engine.calls == [("full", [30])]


def test_extract_results_for_frames_chunks_easyocr_batches():
    engine = FakeOcrEngine(
        {
            1: ("one", 0.8),
            2: ("two", 0.8),
            3: ("three", 0.8),
            4: ("four", 0.8),
            5: ("five", 0.8),
        }
    )
    engine._backend_name = "easyocr"

    results = _extract_results_for_frames(
        ocr_engine=engine,
        frames=[_frame(1), _frame(2), _frame(3), _frame(4), _frame(5)],
        memo={},
    )

    assert results == [
        ("one", 0.8),
        ("two", 0.8),
        ("three", 0.8),
        ("four", 0.8),
        ("five", 0.8),
    ]
    assert engine.calls == [("full", [1, 2, 3, 4]), ("full", [5])]


def test_should_retry_weak_output_flags_single_character_result():
    segments = [TranscriptSegment(start_sec=0.0, end_sec=1.0, text="1", confidence=0.68)]

    assert _should_retry_weak_output(segments) is True


def test_score_ocr_segments_prefers_richer_higher_confidence_text():
    weak_segments = [TranscriptSegment(start_sec=0.0, end_sec=1.0, text="1", confidence=0.68)]
    better_segments = [
        TranscriptSegment(start_sec=0.0, end_sec=1.0, text="付き合いたてに確認したいこと", confidence=0.91),
        TranscriptSegment(start_sec=1.0, end_sec=2.0, text="連絡頻度を決める", confidence=0.88),
    ]

    assert _score_ocr_segments(better_segments) > _score_ocr_segments(weak_segments)


def test_resolve_video_path_rejects_thumbnail_image_urls(tmp_path):
    with pytest.raises(RuntimeError, match="image thumbnail"):
        resolve_video_path(
            "https://i.ytimg.com/vi/abc123/default.jpg",
            download_dir=tmp_path,
            cookies_file=None,
        )
