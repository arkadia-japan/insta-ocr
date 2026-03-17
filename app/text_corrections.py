from __future__ import annotations

import json
import re
from pathlib import Path

from .utils import normalize_text


DEFAULT_CORRECTIONS = {
    "特徹": "特徴",
    "特微": "特徴",
    "短文L1NE": "短文LINE",
    "短文いNE": "短文LINE",
    "距離感がうまい男 ": "距離感がうまい男",
    "自分の時間ある男": "自分の時間がある男",
    "気に入つたら": "気に入ったら",
    "ど一も": "どうも",
    "コオロー": "フォロー",
    "フオロー": "フォロー",
    "フォ口ー": "フォロー",
    "フオロ一": "フォロー",
    "メ気": "※気",
    "糸フォロー": "※フォロー",
}

CTA_FOLLOW_VARIANTS = (
    "フォロー",
    "フオロー",
    "フォ口ー",
    "フオロ一",
    "フォ口一",
    "コオロー",
)


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


def _normalize_cta_line(line: str) -> str:
    compact = re.sub(r"\s+", "", line)
    if "気に入" not in compact:
        return line
    if not any(token in compact for token in CTA_FOLLOW_VARIANTS):
        return line

    compact = compact.lstrip("※＊*")
    compact = compact.replace("メ気", "気")
    compact = compact.replace("※気", "気")
    if not compact.startswith("気に入"):
        return line

    return "※気に入ったらフォロー"


def _normalize_follow_benefit_line(line: str) -> str:
    compact = re.sub(r"\s+", "", line)
    if "恋愛運上がります" not in compact:
        return line
    if "フォ" not in compact and "糸フォ" not in compact:
        return line
    return "※フォローで恋愛運上がります。"


def _normalize_symbol_placeholders(line: str) -> str:
    corrected = line
    if "派かな" in corrected:
        corrected = re.sub(r"[0OＯ〇◯○]{2,}", lambda match: "○" * len(match.group(0)), corrected)
    if "日空いてる" in corrected:
        corrected = re.sub(r"^[0OＯ〇◯○]+(?=日空いてる)", lambda match: "○" * len(match.group(0)), corrected)
        corrected = re.sub(
            r"(?<=[^0-9A-Za-z])[0OＯ〇◯○]+(?=日空いてる)",
            lambda match: "○" * len(match.group(0)),
            corrected,
        )
    return corrected


def _dedupe_corrected_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line:
            if deduped and deduped[-1] != "":
                deduped.append("")
            continue
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    while deduped and deduped[0] == "":
        deduped.pop(0)
    while deduped and deduped[-1] == "":
        deduped.pop()
    return deduped


def _looks_like_title(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    return len(compact) >= 6 and "。" not in compact and "?" not in compact and "？" not in compact


def _looks_like_section_header(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    if not compact or len(compact) > 4:
        return False
    if any(token in compact for token in ("。", "、", "？", "?", "！", "!", ":", "：")):
        return False
    return bool(re.search(r"[A-Za-zぁ-んァ-ヶ一-龠]", compact))


def _looks_like_footer_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    return any(token in compact for token in ("いいね", "保存", "フォロー", "恋愛運"))


def _insert_section_breaks(lines: list[str]) -> list[str]:
    if not lines:
        return []

    formatted: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            if formatted and formatted[-1] != "":
                formatted.append("")
            continue

        previous = next((item for item in reversed(formatted) if item != ""), "")
        should_break = False
        if index == 1 and _looks_like_title(lines[0]) and _looks_like_section_header(line):
            should_break = True
        elif _looks_like_section_header(line) and previous and not _looks_like_section_header(previous):
            should_break = True
        elif index >= 3 and _looks_like_footer_line(line) and previous and not _looks_like_footer_line(previous):
            should_break = True

        if should_break and formatted and formatted[-1] != "":
            formatted.append("")
        formatted.append(line)

    return _dedupe_corrected_lines(formatted)


def apply_text_corrections(text: str, corrections: dict[str, str]) -> str:
    corrected_lines: list[str] = []
    for line in text.splitlines():
        corrected_line = line
        for source, target in sorted(corrections.items(), key=lambda item: len(item[0]), reverse=True):
            if target in corrected_line:
                continue
            corrected_line = corrected_line.replace(source, target)
        corrected_line = _normalize_cta_line(corrected_line)
        corrected_line = _normalize_follow_benefit_line(corrected_line)
        corrected_line = _normalize_symbol_placeholders(corrected_line)
        corrected_lines.append(corrected_line)
    return "\n".join(_insert_section_breaks(_dedupe_corrected_lines(corrected_lines)))
