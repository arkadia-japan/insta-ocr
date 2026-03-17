from app.models import TranscriptSegment, TranscriptionResult


def test_to_full_text_skips_adjacent_duplicate_segments():
    result = TranscriptionResult(
        input_ref="sample",
        platform="instagram",
        video_path="sample.mp4",
        duration_sec=10.0,
        sampled_frames=2,
        ocr_hits=2,
        fallback_used=False,
        segments=[
            TranscriptSegment(start_sec=0.0, end_sec=3.0, text="1行目\n2行目", confidence=0.9),
            TranscriptSegment(start_sec=3.0, end_sec=6.0, text="1行目\n2行目", confidence=0.8),
            TranscriptSegment(start_sec=6.0, end_sec=10.0, text="3行目", confidence=0.7),
        ],
    )

    assert result.to_full_text() == "1行目\n2行目\n\n3行目"
    assert result.to_dict()["transcription_mode"] == "ocr"


def test_to_full_text_with_timestamps_includes_time_range():
    result = TranscriptionResult(
        input_ref="sample",
        platform="instagram",
        video_path="sample.mp4",
        duration_sec=5.0,
        sampled_frames=1,
        ocr_hits=1,
        fallback_used=False,
        segments=[
            TranscriptSegment(start_sec=0.0, end_sec=5.0, text="全文", confidence=0.9),
        ],
    )

    assert result.to_full_text(include_timestamps=True) == "[00:00:00.00 - 00:00:05.00]\n全文"
