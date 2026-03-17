from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

from .utils import ensure_directory


def _ascii_safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "file"


def _candidate_runtime_roots() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("SHORTS_VISUAL_TRANSCRIBER_RUNTIME", "").strip()
    if configured:
        candidates.append(Path(configured))

    public_dir = os.environ.get("PUBLIC", r"C:\Users\Public")
    candidates.append(Path(public_dir) / "shorts_visual_transcriber_runtime")
    candidates.append(Path(r"C:\shorts_visual_transcriber_runtime"))
    return candidates


def get_runtime_root() -> Path:
    last_error: OSError | None = None
    for candidate in _candidate_runtime_roots():
        try:
            ensure_directory(candidate)
            return candidate.resolve()
        except OSError as exc:
            last_error = exc
    raise RuntimeError("ASCII runtime directory could not be created.") from last_error


def get_runtime_download_dir() -> Path:
    path = get_runtime_root() / "downloads"
    ensure_directory(path)
    return path


def get_runtime_output_dir() -> Path:
    path = get_runtime_root() / "output"
    ensure_directory(path)
    return path


def get_runtime_upload_dir() -> Path:
    path = get_runtime_root() / "uploads"
    ensure_directory(path)
    return path


def get_runtime_paddlex_cache_dir() -> Path:
    path = get_runtime_root() / "paddlex_cache"
    ensure_directory(path)
    return path


def get_runtime_paddleocr_dir() -> Path:
    path = get_runtime_root() / "paddleocr"
    ensure_directory(path)
    return path


def configure_paddle_runtime_env() -> dict[str, str]:
    settings = {
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "PADDLE_PDX_CACHE_HOME": str(get_runtime_paddlex_cache_dir()),
        "PADDLE_OCR_BASE_DIR": str(get_runtime_paddleocr_dir()),
    }
    for key, value in settings.items():
        os.environ[key] = value
    return settings


def is_ascii_path(path: Path | str) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def stage_video_for_runtime(video_path: Path, runtime_video_dir: Path) -> Path:
    source = video_path.expanduser().resolve()
    ensure_directory(runtime_video_dir)
    if is_ascii_path(source):
        return source

    target = runtime_video_dir / f"{uuid4().hex}_{_ascii_safe_name(source.name)}"
    shutil.copy2(source, target)
    return target.resolve()
