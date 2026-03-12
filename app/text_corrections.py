from __future__ import annotations

import json
from pathlib import Path


DEFAULT_CORRECTIONS = {
    "特微": "特徴",
    "特徹": "特徴",
    "亜離感": "距離感",
    "跳離感": "距離感",
    "短文いNE": "短文LINE",
    "※気に入ったらフォローして": "※気に入ったらフォローしてね!",
}


def load_text_corrections(
    project_root: Path | None = None,
    correction_file: Path | None = None,
) -> dict[str, str]:
    base_dir = project_root or Path(__file__).resolve().parents[1]
    correction_path = correction_file or (base_dir / "ocr_corrections.json")
    corrections = dict(DEFAULT_CORRECTIONS)

    if correction_path.exists():
        payload = json.loads(correction_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            corrections.update({str(key): str(value) for key, value in payload.items()})

    return corrections


def apply_text_corrections(text: str, corrections: dict[str, str]) -> str:
    corrected_lines: list[str] = []
    for line in text.splitlines():
        corrected_line = line
        for source, target in sorted(corrections.items(), key=lambda item: len(item[0]), reverse=True):
            if target in corrected_line:
                continue
            corrected_line = corrected_line.replace(source, target)
        corrected_lines.append(corrected_line)
    return "\n".join(corrected_lines)
