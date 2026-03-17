from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .runtime_paths import get_runtime_config_dir

RUN_MODE_PENDING = "pending"
RUN_MODE_TEST_ONE = "test_one"
RUN_MODE_RETRY_ERRORS = "retry_errors"
RUN_MODE_RERUN_IMAGES = "rerun_images"
RUN_MODE_RERUN_AUDIO = "rerun_audio"


@dataclass(frozen=True)
class SheetSyncPreset:
    name: str
    service_account_file: str
    spreadsheet_id: str
    sheet_name: str
    url_column: str = "G"
    style_column: str = "K"
    transcription_column: str = "L"
    status_column: str = "M"
    processed_at_column: str = "N"
    error_column: str = "O"
    audio_style: str = "音声リール"
    image_style: str = "画像リール"
    google_python: str = r"C:\Python313\python.exe"


@dataclass(frozen=True)
class SheetSyncRunRequest:
    mode: str
    output_dir: str
    download_dir: str
    cookies_file: str = ""
    audio_model_size: str = "small"
    audio_language: str = "ja"
    langs: str = "ja,en"
    min_ocr_confidence: float = 0.15
    gpu: bool = False
    sample_fps: float = 4.0
    scene_threshold: float = 0.22
    text_change_threshold: float = 0.055
    min_interval_sec: float = 0.4
    similarity_threshold: float = 0.86
    keep_video: bool = False


def default_sheet_sync_preset() -> SheetSyncPreset:
    return SheetSyncPreset(
        name="現在の運用設定",
        service_account_file=os.environ.get(
            "SHEETS_SYNC_SERVICE_ACCOUNT_FILE",
            r"C:\Users\yoona\secure\shorts-analyzer-490407-b7632627fff3.json",
        ),
        spreadsheet_id=os.environ.get(
            "SHEETS_SYNC_SPREADSHEET_ID",
            "1kFFRfOkcgtp0a4SZZ5q5hxMXLNAwlAGO6-bCEROAgCo",
        ),
        sheet_name=os.environ.get("SHEETS_SYNC_SHEET_NAME", "投稿データ260301"),
        url_column=os.environ.get("SHEETS_SYNC_URL_COLUMN", "G"),
        google_python=os.environ.get("SHEETS_SYNC_GOOGLE_PYTHON", r"C:\Python313\python.exe"),
    )


def get_sheet_dashboard_config_path(config_dir: Path | None = None) -> Path:
    base_dir = config_dir or get_runtime_config_dir()
    return base_dir / "sheet_sync_presets.json"


def load_sheet_sync_presets(
    config_path: Path | None = None,
) -> tuple[list[SheetSyncPreset], str]:
    path = config_path or get_sheet_dashboard_config_path()
    default_preset = default_sheet_sync_preset()

    if not path.exists():
        return [default_preset], default_preset.name

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_presets = payload.get("presets", []) if isinstance(payload, dict) else []
    presets = [_coerce_preset(item) for item in raw_presets if isinstance(item, dict)]
    presets = [preset for preset in presets if preset.name.strip()]
    if not presets:
        return [default_preset], default_preset.name

    selected_name = payload.get("selected_name", presets[0].name) if isinstance(payload, dict) else presets[0].name
    if selected_name not in {preset.name for preset in presets}:
        selected_name = presets[0].name
    return presets, selected_name


def save_sheet_sync_presets(
    presets: list[SheetSyncPreset],
    selected_name: str,
    config_path: Path | None = None,
) -> Path:
    path = config_path or get_sheet_dashboard_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_name": selected_name,
        "presets": [asdict(preset) for preset in presets],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_sheet_sync_command(
    python_executable: str,
    preset: SheetSyncPreset,
    request: SheetSyncRunRequest,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "app.sheets_sync",
        "--service-account-file",
        preset.service_account_file,
        "--spreadsheet-id",
        preset.spreadsheet_id,
        "--sheet-name",
        preset.sheet_name,
        "--url-column",
        preset.url_column,
        "--style-column",
        preset.style_column,
        "--transcription-column",
        preset.transcription_column,
        "--status-column",
        preset.status_column,
        "--processed-at-column",
        preset.processed_at_column,
        "--error-column",
        preset.error_column,
        "--audio-style",
        preset.audio_style,
        "--image-style",
        preset.image_style,
        "--output-dir",
        request.output_dir,
        "--download-dir",
        request.download_dir,
        "--audio-model-size",
        request.audio_model_size,
        "--audio-language",
        request.audio_language,
        "--langs",
        request.langs,
        "--min-ocr-confidence",
        f"{request.min_ocr_confidence:.2f}",
        "--sample-fps",
        f"{request.sample_fps:.2f}",
        "--scene-threshold",
        f"{request.scene_threshold:.3f}",
        "--text-change-threshold",
        f"{request.text_change_threshold:.3f}",
        "--min-interval-sec",
        f"{request.min_interval_sec:.2f}",
        "--similarity-threshold",
        f"{request.similarity_threshold:.2f}",
    ]

    if request.cookies_file.strip():
        command.extend(["--cookies-file", request.cookies_file.strip()])
    if request.keep_video:
        command.append("--keep-video")
    if request.gpu:
        command.append("--gpu")

    if request.mode == RUN_MODE_TEST_ONE:
        command.extend(["--limit", "1"])
    elif request.mode == RUN_MODE_RETRY_ERRORS:
        command.append("--retry-errors")
    elif request.mode == RUN_MODE_RERUN_IMAGES:
        command.extend(["--only-style", preset.image_style, "--reprocess-existing"])
    elif request.mode == RUN_MODE_RERUN_AUDIO:
        command.extend(["--only-style", preset.audio_style, "--reprocess-existing"])

    return command


def summarize_sheet_sync_output(output: str) -> dict[str, int]:
    text = output or ""
    return {
        "rows": len(re.findall(r"(?m)^\[INFO] row=", text)),
        "done": len(re.findall(r"(?m)^\[DONE]", text)),
        "errors": len(re.findall(r"(?m)^\[ERROR]", text)),
    }


def read_text_file_best_effort(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _coerce_preset(payload: dict[str, str]) -> SheetSyncPreset:
    default = default_sheet_sync_preset()
    values = asdict(default)
    for key in values:
        if key in payload:
            values[key] = str(payload[key])
    return SheetSyncPreset(**values)
