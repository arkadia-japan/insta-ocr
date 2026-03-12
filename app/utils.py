from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def detect_platform(value: str) -> str:
    lowered = value.lower()
    if "tiktok.com" in lowered:
        return "tiktok"
    if "instagram.com" in lowered:
        return "instagram"
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    return "unknown"


def safe_stem_from_input(value: str) -> str:
    if is_url(value):
        parsed = urlparse(value)
        raw = f"{parsed.netloc}_{parsed.path.strip('/') or 'video'}"
    else:
        raw = Path(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return (cleaned or "transcript")[:80]


def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    cjk = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f"
    compact = re.sub(rf"(?<=[{cjk}])\s+(?=[{cjk}])", "", compact)
    compact = re.sub(rf"(?<=[{cjk}])\s+(?=[!！?？:：;；,，.．])", "", compact)
    compact = re.sub(rf"(?<=[!！?？:：;；,，.．])\s+(?=[{cjk}])", "", compact)
    return compact


def format_compact_time(seconds: float) -> str:
    clamped = max(0.0, float(seconds))
    hours = int(clamped // 3600)
    minutes = int((clamped % 3600) // 60)
    secs = clamped % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def format_srt_time(seconds: float) -> str:
    clamped = max(0.0, float(seconds))
    milliseconds_total = int(round(clamped * 1000))
    hours = milliseconds_total // 3_600_000
    minutes = (milliseconds_total % 3_600_000) // 60_000
    secs = (milliseconds_total % 60_000) // 1000
    millis = milliseconds_total % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
