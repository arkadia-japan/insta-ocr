from __future__ import annotations

import json
from pathlib import Path

from .models import TranscriptSegment, TranscriptionResult
from .utils import ensure_directory, format_compact_time, format_srt_time


def _write_json(result: TranscriptionResult, output_path: Path, json_indent: int = 2) -> None:
    payload = result.to_dict()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=json_indent),
        encoding="utf-8",
    )


def _write_txt(segments: list[TranscriptSegment], output_path: Path) -> None:
    lines: list[str] = []
    for segment in segments:
        start = format_compact_time(segment.start_sec)
        end = format_compact_time(segment.end_sec)
        lines.append(f"[{start} - {end}] {segment.text}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_srt(segments: list[TranscriptSegment], output_path: Path) -> None:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_time(segment.start_sec)
        end = format_srt_time(segment.end_sec)
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text}\n")
    output_path.write_text("\n".join(blocks), encoding="utf-8")


def write_outputs(
    result: TranscriptionResult,
    output_dir: Path,
    stem: str,
    json_indent: int = 2,
) -> dict[str, Path]:
    ensure_directory(output_dir)
    json_path = output_dir / f"{stem}.json"
    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"

    _write_json(result, json_path, json_indent=json_indent)
    _write_txt(result.segments, txt_path)
    _write_srt(result.segments, srt_path)

    return {"json": json_path, "txt": txt_path, "srt": srt_path}
