from __future__ import annotations

from difflib import SequenceMatcher

from .models import Snapshot, TranscriptSegment
from .utils import normalize_text


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def merge_snapshots_to_segments(
    snapshots: list[Snapshot],
    duration_sec: float,
    similarity_threshold: float = 0.86,
    min_segment_sec: float = 0.5,
) -> list[TranscriptSegment]:
    if not snapshots:
        return []

    ordered = sorted(snapshots, key=lambda item: item.time_sec)
    segment_groups: list[dict] = []

    for index, snapshot in enumerate(ordered):
        if not snapshot.text:
            continue
        next_time = ordered[index + 1].time_sec if (index + 1) < len(ordered) else duration_sec
        next_time = max(next_time, snapshot.time_sec + min_segment_sec)

        if segment_groups and _similarity(segment_groups[-1]["seed_text"], snapshot.text) >= similarity_threshold:
            segment_groups[-1]["end_sec"] = max(segment_groups[-1]["end_sec"], next_time)
            segment_groups[-1]["snapshots"].append(snapshot)
            continue

        segment_groups.append(
            {
                "start_sec": snapshot.time_sec,
                "end_sec": next_time,
                "seed_text": snapshot.text,
                "snapshots": [snapshot],
            }
        )

    segments: list[TranscriptSegment] = []
    for group in segment_groups:
        text = _build_consensus_text(group["snapshots"])
        confidence = _average_confidence(group["snapshots"])
        segments.append(
            TranscriptSegment(
                start_sec=group["start_sec"],
                end_sec=group["end_sec"],
                text=text,
                confidence=confidence,
            )
        )

    for idx in range(1, len(segments)):
        previous = segments[idx - 1]
        current = segments[idx]
        if current.start_sec < previous.end_sec:
            current.start_sec = previous.end_sec
        if current.end_sec < current.start_sec + min_segment_sec:
            current.end_sec = current.start_sec + min_segment_sec

    for segment in segments:
        segment.start_sec = max(0.0, segment.start_sec)
        segment.end_sec = max(segment.start_sec + min_segment_sec, segment.end_sec)
        if duration_sec > 0:
            segment.start_sec = min(segment.start_sec, duration_sec)
            segment.end_sec = min(segment.end_sec, duration_sec)

    return [segment for segment in segments if segment.text.strip()]


def harmonize_segment_lines(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not segments:
        return []

    occurrences: list[dict] = []
    for segment_index, segment in enumerate(segments):
        lines = [normalize_text(line) for line in segment.text.splitlines() if normalize_text(line)]
        for line_index, line in enumerate(lines):
            occurrences.append(
                {
                    "segment_index": segment_index,
                    "line_index": line_index,
                    "text": line,
                    "confidence": segment.confidence or 0.0,
                }
            )

    groups: list[list[dict]] = []
    for occurrence in occurrences:
        matched_group: list[dict] | None = None
        for group in groups:
            if any(item["segment_index"] == occurrence["segment_index"] for item in group):
                continue
            if _should_group_lines(occurrence["text"], group[0]["text"]):
                matched_group = group
                break
        if matched_group is None:
            groups.append([occurrence])
        else:
            matched_group.append(occurrence)

    best_text_by_key: dict[tuple[int, int], str] = {}
    for group in groups:
        best = max(
            group,
            key=lambda item: (
                _line_quality_score(item["text"]) + item["confidence"] * 12.0,
                item["confidence"],
                len(item["text"]),
            ),
        )
        for item in group:
            best_text_by_key[(item["segment_index"], item["line_index"])] = best["text"]

    harmonized: list[TranscriptSegment] = []
    for segment_index, segment in enumerate(segments):
        raw_lines = segment.text.splitlines()
        lines = [normalize_text(line) for line in raw_lines if normalize_text(line)]
        updated_lines = [
            best_text_by_key.get((segment_index, line_index), line)
            for line_index, line in enumerate(lines)
        ]
        reconstructed_lines: list[str] = []
        updated_index = 0
        for raw_line in raw_lines:
            normalized = normalize_text(raw_line)
            if not normalized:
                reconstructed_lines.append("")
                continue
            reconstructed_lines.append(updated_lines[updated_index])
            updated_index += 1
        harmonized.append(
            TranscriptSegment(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text=_dedupe_text_lines("\n".join(reconstructed_lines)),
                confidence=segment.confidence,
            )
        )
    return harmonized


def consolidate_ocr_candidates(candidates: list[tuple[str, float | None]]) -> tuple[str, float | None]:
    snapshots = [
        Snapshot(time_sec=float(index), text=text, confidence=confidence)
        for index, (text, confidence) in enumerate(candidates)
        if normalize_text(text)
    ]
    if not snapshots:
        return "", None
    if len(snapshots) == 1:
        return _dedupe_text_lines(snapshots[0].text), snapshots[0].confidence

    groups: list[list[Snapshot]] = []
    for snapshot in snapshots:
        matched_group: list[Snapshot] | None = None
        for group in groups:
            if _text_similarity(snapshot.text, group[0].text) >= 0.68:
                matched_group = group
                break
        if matched_group is None:
            groups.append([snapshot])
        else:
            matched_group.append(snapshot)

    best_group = max(
        groups,
        key=lambda group: (
            len(group),
            max(_text_quality_score(item.text, item.confidence or 0.0) for item in group),
            sum(_text_quality_score(item.text, item.confidence or 0.0) for item in group),
            sum(item.confidence or 0.0 for item in group),
        ),
    )
    consensus_text = _build_consensus_text(best_group)
    if not consensus_text:
        chosen = max(
            best_group,
            key=lambda item: (
                _text_quality_score(item.text, item.confidence or 0.0),
                item.confidence or 0.0,
                len(normalize_text(item.text)),
            ),
        )
        consensus_text = chosen.text
    return _dedupe_text_lines(consensus_text), _average_confidence(best_group)


def merge_adjacent_similar_segments(
    segments: list[TranscriptSegment],
    similarity_threshold: float = 0.84,
) -> list[TranscriptSegment]:
    if not segments:
        return []

    merged_groups: list[list[TranscriptSegment]] = [[segments[0]]]
    for segment in segments[1:]:
        current_group = merged_groups[-1]
        if _should_merge_adjacent_segments(
            current_group[-1],
            segment,
            similarity_threshold=similarity_threshold,
        ):
            current_group.append(segment)
        else:
            merged_groups.append([segment])

    merged_segments: list[TranscriptSegment] = []
    for group in merged_groups:
        text, confidence = consolidate_ocr_candidates([(item.text, item.confidence) for item in group])
        if not text:
            continue
        merged_segments.append(
            TranscriptSegment(
                start_sec=group[0].start_sec,
                end_sec=group[-1].end_sec,
                text=text,
                confidence=confidence,
            )
        )
    return merged_segments


def _build_consensus_text(snapshots: list[Snapshot]) -> str:
    if len(snapshots) == 1:
        return snapshots[0].text

    per_snapshot_lines = [
        [normalize_text(line) for line in snapshot.text.splitlines() if normalize_text(line)]
        for snapshot in snapshots
    ]
    max_lines = max((len(lines) for lines in per_snapshot_lines), default=0)
    consensus_lines: list[str] = []

    for index in range(max_lines):
        line_candidates: list[tuple[str, float]] = []
        for snapshot, lines in zip(snapshots, per_snapshot_lines):
            if index < len(lines):
                line_candidates.append((lines[index], snapshot.confidence or 0.0))

        best_line = _choose_consensus_line(line_candidates)
        if best_line:
            consensus_lines.append(best_line)

    return "\n".join(consensus_lines)


def _choose_consensus_line(candidates: list[tuple[str, float]]) -> str:
    if not candidates:
        return ""

    groups: list[list[tuple[str, float]]] = []
    for candidate in candidates:
        matched_group: list[tuple[str, float]] | None = None
        for group in groups:
            if _line_similarity(candidate[0], group[0][0]) >= 0.72:
                matched_group = group
                break
        if matched_group is None:
            groups.append([candidate])
        else:
            matched_group.append(candidate)

    def group_score(group: list[tuple[str, float]]) -> tuple[int, int, float, int]:
        return (
            len(group),
            max(int(round(_line_quality_score(text))) for text, _ in group),
            sum(conf for _, conf in group),
            max(len(text) for text, _ in group),
        )

    best_group = max(groups, key=group_score)
    return max(
        best_group,
        key=lambda item: (_line_quality_score(item[0]) + item[1] * 15.0, item[1], len(item[0])),
    )[0]


def _line_similarity(left: str, right: str) -> float:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _text_similarity(left: str, right: str) -> float:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _should_merge_adjacent_segments(
    left: TranscriptSegment,
    right: TranscriptSegment,
    similarity_threshold: float,
) -> bool:
    similarity = _text_similarity(left.text, right.text)
    if similarity >= similarity_threshold:
        return True

    overlap_ratio = _line_overlap_ratio(left.text, right.text)
    confidence_gap = (right.confidence or 0.0) - (left.confidence or 0.0)
    shorter_is_noisier = _line_noise_count(left.text) >= _line_noise_count(right.text)
    if overlap_ratio >= 0.6 and (confidence_gap >= -0.05 or shorter_is_noisier):
        return True
    return _should_absorb_short_intro(left, right)


def _line_overlap_ratio(left: str, right: str) -> float:
    left_lines = {normalize_text(line) for line in left.splitlines() if normalize_text(line)}
    right_lines = {normalize_text(line) for line in right.splitlines() if normalize_text(line)}
    if not left_lines or not right_lines:
        return 0.0
    intersection = len(left_lines & right_lines)
    return intersection / max(1, min(len(left_lines), len(right_lines)))


def _should_absorb_short_intro(left: TranscriptSegment, right: TranscriptSegment) -> bool:
    duration = max(0.0, left.end_sec - left.start_sec)
    if duration > 0.8:
        return False

    left_lines = [normalize_text(line) for line in left.text.splitlines() if normalize_text(line)]
    right_lines = [normalize_text(line) for line in right.text.splitlines() if normalize_text(line)]
    if not left_lines or not right_lines:
        return False

    title_similarity = _text_similarity(left_lines[0], right_lines[0])
    left_length = len(normalize_text(left.text))
    right_length = len(normalize_text(right.text))
    confidence_gap = (right.confidence or 0.0) - (left.confidence or 0.0)
    return (
        title_similarity >= 0.72
        and left_length <= max(12, int(right_length * 0.65))
        and confidence_gap >= -0.02
    )


def _average_confidence(snapshots: list[Snapshot]) -> float | None:
    confidences = [snapshot.confidence for snapshot in snapshots if snapshot.confidence is not None]
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)


def _dedupe_text_lines(text: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _text_noise_count(text: str) -> int:
    return sum(_line_noise_count(line) for line in text.splitlines())


def _text_quality_score(text: str, confidence: float) -> float:
    lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
    if not lines:
        return -999.0

    single_char_lines = sum(1 for line in lines if len(line.replace(" ", "")) <= 1)
    return (
        confidence * 100.0
        + sum(_line_quality_score(line) for line in lines)
        + len(lines) * 4.0
        - single_char_lines * 12.0
        - _text_noise_count(text) * 8.0
    )


def _line_noise_count(text: str) -> int:
    return sum(char in "|[]{}<>~`" for char in text) + sum(
        char.isascii() and not (char.isalnum() or char.isspace() or char in "!?.,:;/-_()#%&'\"")
        for char in text
    )


def _line_quality_score(text: str) -> float:
    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    if not compact:
        return -999.0

    japanese_chars = _count_japanese_chars(compact)
    ascii_chars = sum(char.isascii() for char in compact)
    digit_chars = sum(char.isdigit() for char in compact)
    noise_count = _line_noise_count(normalized)
    ascii_ratio = ascii_chars / max(len(compact), 1)

    score = min(len(compact), 42) * 1.3 + japanese_chars * 3.4 - noise_count * 14.0
    if len(compact) <= 1:
        score -= 20.0
    if digit_chars == len(compact) and len(compact) <= 4:
        score -= 24.0
    if ascii_chars == len(compact) and len(compact) <= 4:
        score -= 12.0
    if ascii_ratio >= 0.7 and japanese_chars == 0 and len(compact) <= 8:
        score -= 10.0
    return score


def _count_japanese_chars(text: str) -> int:
    count = 0
    for char in text:
        codepoint = ord(char)
        if (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xFF66 <= codepoint <= 0xFF9F
        ):
            count += 1
    return count


def _should_group_lines(left: str, right: str) -> bool:
    similarity = _line_similarity(left, right)
    if similarity >= 0.72:
        return True
    if similarity < 0.58:
        return False
    return _line_noise_count(left) > 0 or _line_noise_count(right) > 0
