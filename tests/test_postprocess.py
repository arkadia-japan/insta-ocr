from app.models import Snapshot, TranscriptSegment
from app.postprocess import (
    consolidate_ocr_candidates,
    harmonize_segment_lines,
    merge_adjacent_similar_segments,
    merge_snapshots_to_segments,
)


def test_merge_snapshots_combines_similar_text():
    snapshots = [
        Snapshot(time_sec=0.0, text="Hello world", confidence=0.9),
        Snapshot(time_sec=1.0, text="Hello world!", confidence=0.8),
        Snapshot(time_sec=2.0, text="Different text", confidence=0.7),
    ]

    segments = merge_snapshots_to_segments(
        snapshots=snapshots,
        duration_sec=3.0,
        similarity_threshold=0.8,
    )

    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[1].text == "Different text"


def test_merge_snapshots_empty_input():
    assert merge_snapshots_to_segments([], duration_sec=5.0) == []


def test_merge_snapshots_builds_consensus_text_across_frames():
    snapshots = [
        Snapshot(time_sec=0.0, text="Top line\nSecond line", confidence=0.6),
        Snapshot(time_sec=0.5, text="Top line\nSecond line", confidence=0.9),
        Snapshot(time_sec=1.0, text="Top line\nSecond liNe", confidence=0.7),
    ]

    segments = merge_snapshots_to_segments(
        snapshots=snapshots,
        duration_sec=2.0,
        similarity_threshold=0.7,
    )

    assert len(segments) == 1
    assert segments[0].text == "Top line\nSecond line"


def test_harmonize_segment_lines_propagates_cleaner_repeated_line():
    harmonized = harmonize_segment_lines(
        [
            TranscriptSegment(
                start_sec=0.0,
                end_sec=1.0,
                text="Save it\nThen follow this account!",
                confidence=0.9,
            ),
            TranscriptSegment(
                start_sec=1.0,
                end_sec=2.0,
                text="Save it\nThen follow this account |",
                confidence=0.7,
            ),
        ]
    )

    assert harmonized[1].text == "Save it\nThen follow this account!"


def test_harmonize_segment_lines_does_not_merge_different_clean_lines():
    harmonized = harmonize_segment_lines(
        [
            TranscriptSegment(
                start_sec=0.0,
                end_sec=1.0,
                text="Independent man",
                confidence=0.9,
            ),
            TranscriptSegment(
                start_sec=1.0,
                end_sec=2.0,
                text="Good at keeping distance",
                confidence=0.8,
            ),
        ]
    )

    assert harmonized[0].text == "Independent man"
    assert harmonized[1].text == "Good at keeping distance"


def test_harmonize_segment_lines_does_not_merge_similar_lines_inside_one_segment():
    harmonized = harmonize_segment_lines(
        [
            TranscriptSegment(
                start_sec=0.0,
                end_sec=2.0,
                text="Has his own axis\nHas his own time",
                confidence=0.9,
            )
        ]
    )

    assert harmonized[0].text == "Has his own axis\nHas his own time"


def test_consolidate_ocr_candidates_prefers_consensus():
    text, confidence = consolidate_ocr_candidates(
        [
            ("Top line\nShort LINE", 0.6),
            ("Top line\nShort LINE", 0.8),
            ("Top line\nShort L1NE", 0.7),
        ]
    )

    assert text == "Top line\nShort LINE"
    assert confidence == 0.7


def test_merge_adjacent_similar_segments_collapses_duplicates():
    segments = merge_adjacent_similar_segments(
        [
            TranscriptSegment(start_sec=0.0, end_sec=0.5, text="Top line", confidence=0.4),
            TranscriptSegment(start_sec=0.5, end_sec=2.0, text="Top line", confidence=0.9),
            TranscriptSegment(start_sec=2.0, end_sec=3.0, text="Different slide", confidence=0.8),
        ]
    )

    assert len(segments) == 2
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 2.0
    assert segments[0].text == "Top line"


def test_merge_adjacent_similar_segments_absorbs_partial_intro_text():
    segments = merge_adjacent_similar_segments(
        [
            TranscriptSegment(
                start_sec=0.0,
                end_sec=0.5,
                text="Title\nLine A\nLine B\nCall to action",
                confidence=0.45,
            ),
            TranscriptSegment(
                start_sec=0.5,
                end_sec=3.0,
                text="Title\nLine A\nLine B\nLine C\nLine D\nCall to action",
                confidence=0.82,
            ),
        ]
    )

    assert len(segments) == 1
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 3.0


def test_merge_adjacent_similar_segments_absorbs_short_title_matched_intro():
    segments = merge_adjacent_similar_segments(
        [
            TranscriptSegment(
                start_sec=0.0,
                end_sec=0.4,
                text="Slide title\nPartial text",
                confidence=0.55,
            ),
            TranscriptSegment(
                start_sec=0.4,
                end_sec=2.5,
                text="Slide title\nFull line one\nFull line two\nCall to action",
                confidence=0.8,
            ),
        ]
    )

    assert len(segments) == 1
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 2.5
