from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import ensure_directory

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class DownloadError(RuntimeError):
    pass


def _collect_video_files(directory: Path) -> dict[Path, float]:
    entries: dict[Path, float] = {}
    for path in directory.glob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        entries[path.resolve()] = path.stat().st_mtime
    return entries


def _pick_downloaded_file(before: dict[Path, float], after: dict[Path, float]) -> Path | None:
    candidates: list[Path] = []
    for path, mtime in after.items():
        old_mtime = before.get(path)
        if old_mtime is None or mtime > old_mtime:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: after[path])


def download_video(
    url: str,
    download_dir: Path,
    cookies_file: Path | None = None,
    ytdlp_binary: str = "yt-dlp",
) -> Path:
    ensure_directory(download_dir)
    before = _collect_video_files(download_dir)

    output_template = str(download_dir / "%(extractor)s_%(id)s.%(ext)s")
    command = [
        ytdlp_binary,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        "--print",
        "after_move:filepath",
        url,
    ]
    if cookies_file:
        command[1:1] = ["--cookies", str(cookies_file)]

    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise DownloadError(
            "yt-dlp failed.\n"
            f"URL: {url}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    after = _collect_video_files(download_dir)
    downloaded = _pick_downloaded_file(before, after)
    if downloaded:
        return downloaded

    # Fallback: inspect path-like lines in stdout.
    for line in reversed((completed.stdout or "").splitlines()):
        candidate = Path(line.strip().strip('"'))
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise DownloadError(
        "yt-dlp completed but no output video file was detected.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
