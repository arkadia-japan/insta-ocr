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
RUN_MODE_RERUN_LOW_QUALITY_OCR = "rerun_low_quality_ocr"
MANAGED_SHEET_SYNC_SPREADSHEETS = (
    {
        "name": "インスタ運用シート",
        "spreadsheet_label": "インスタ運用シート",
        "spreadsheet_id": "11qZWfM1fB-tY5KRUBmK0YJXX_xyUGU32-XgGI07oS0k",
    },
    {
        "name": "Shorts運用シート",
        "spreadsheet_label": "Shorts運用シート",
        "spreadsheet_id": "1c4zJfsspMXZPMoMPeqnNujV3qqapDx4LMZQ7JOucUT4",
    },
    {
        "name": "Tiktok運用シート",
        "spreadsheet_label": "Tiktok運用シート",
        "spreadsheet_id": "1z3en3A7b1NzUJ7kTyknYBWXXsXWBVPQf2OTrkoJ-DgE",
    },
)


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
    spreadsheet_label: str = ""


@dataclass(frozen=True)
class SheetSyncRunRequest:
    mode: str
    output_dir: str
    download_dir: str
    cookies_file: str = ""
    audio_model_size: str = "small"
    audio_language: str = "ja"
    langs: str = "ja,en"
    ocr_backend: str = "auto"
    min_ocr_confidence: float = 0.15
    gpu: bool = False
    sample_fps: float = 4.0
    scene_threshold: float = 0.22
    text_change_threshold: float = 0.055
    min_interval_sec: float = 0.4
    similarity_threshold: float = 0.86
    keep_video: bool = False


def resolve_sheet_sync_ocr_backend(request: SheetSyncRunRequest) -> str:
    if request.mode == RUN_MODE_RERUN_LOW_QUALITY_OCR:
        return "easyocr"
    if request.ocr_backend == "easyocr" and request.mode != RUN_MODE_TEST_ONE:
        return "paddle"
    return request.ocr_backend


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
    default_presets = merge_managed_sheet_sync_presets([default_preset])

    if not path.exists():
        return default_presets, default_preset.name

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_presets = payload.get("presets", []) if isinstance(payload, dict) else []
    presets = [_coerce_preset(item) for item in raw_presets if isinstance(item, dict)]
    presets = [preset for preset in presets if preset.name.strip()]
    if not presets:
        return default_presets, default_preset.name

    presets = merge_managed_sheet_sync_presets(presets)

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


def get_sheet_sync_spreadsheet_label(preset: SheetSyncPreset) -> str:
    if preset.spreadsheet_label.strip():
        return preset.spreadsheet_label.strip()
    for managed in MANAGED_SHEET_SYNC_SPREADSHEETS:
        if preset.spreadsheet_id == managed["spreadsheet_id"]:
            return managed["spreadsheet_label"]
    if preset.name.strip():
        return preset.name.strip()
    if preset.spreadsheet_id.strip():
        return preset.spreadsheet_id.strip()
    return "未設定"


def build_managed_sheet_sync_presets(base_preset: SheetSyncPreset | None = None) -> list[SheetSyncPreset]:
    base = base_preset or default_sheet_sync_preset()
    presets: list[SheetSyncPreset] = []
    for managed in MANAGED_SHEET_SYNC_SPREADSHEETS:
        presets.append(
            SheetSyncPreset(
                name=managed["name"],
                service_account_file=base.service_account_file,
                spreadsheet_id=managed["spreadsheet_id"],
                sheet_name="",
                url_column=base.url_column,
                style_column=base.style_column,
                transcription_column=base.transcription_column,
                status_column=base.status_column,
                processed_at_column=base.processed_at_column,
                error_column=base.error_column,
                audio_style=base.audio_style,
                image_style=base.image_style,
                google_python=base.google_python,
                spreadsheet_label=managed["spreadsheet_label"],
            )
        )
    return presets


def merge_managed_sheet_sync_presets(presets: list[SheetSyncPreset]) -> list[SheetSyncPreset]:
    merged: list[SheetSyncPreset] = []
    seen_spreadsheet_ids: set[str] = set()
    seen_names: set[str] = set()

    for preset in presets:
        normalized = _normalize_sheet_sync_preset(preset)
        merged.append(normalized)
        seen_spreadsheet_ids.add(normalized.spreadsheet_id)
        seen_names.add(normalized.name)

    for managed_preset in build_managed_sheet_sync_presets():
        if managed_preset.spreadsheet_id in seen_spreadsheet_ids:
            continue
        candidate_name = managed_preset.name
        suffix = 2
        while candidate_name in seen_names:
            candidate_name = f"{managed_preset.name} {suffix}"
            suffix += 1
        normalized = SheetSyncPreset(
            name=candidate_name,
            service_account_file=managed_preset.service_account_file,
            spreadsheet_id=managed_preset.spreadsheet_id,
            sheet_name=managed_preset.sheet_name,
            url_column=managed_preset.url_column,
            style_column=managed_preset.style_column,
            transcription_column=managed_preset.transcription_column,
            status_column=managed_preset.status_column,
            processed_at_column=managed_preset.processed_at_column,
            error_column=managed_preset.error_column,
            audio_style=managed_preset.audio_style,
            image_style=managed_preset.image_style,
            google_python=managed_preset.google_python,
            spreadsheet_label=managed_preset.spreadsheet_label,
        )
        merged.append(normalized)
        seen_spreadsheet_ids.add(normalized.spreadsheet_id)
        seen_names.add(normalized.name)

    return merged


def upsert_sheet_sync_preset(
    preset_map: dict[str, SheetSyncPreset],
    preset: SheetSyncPreset,
    *,
    previous_name: str = "",
) -> dict[str, SheetSyncPreset]:
    next_name = preset.name.strip()
    current_name = previous_name.strip()

    if not next_name:
        raise ValueError("Preset name is required.")
    if next_name in preset_map and next_name != current_name:
        raise ValueError(f"A preset named '{next_name}' already exists.")

    if not current_name or current_name not in preset_map:
        updated_map = dict(preset_map)
        updated_map[next_name] = preset
        return updated_map

    updated_map: dict[str, SheetSyncPreset] = {}
    for name, current_preset in preset_map.items():
        if name == current_name:
            updated_map[next_name] = preset
            continue
        updated_map[name] = current_preset
    return updated_map


def resolve_sheet_sync_selected_preset(
    preset_options: list[str],
    *,
    selected_name: str = "",
    widget_value: str = "",
    pending_name: str = "",
) -> str:
    for candidate in (pending_name, widget_value, selected_name):
        if candidate in preset_options:
            return candidate
    return preset_options[0] if preset_options else ""


def group_sheet_sync_presets(presets: list[SheetSyncPreset]) -> dict[str, list[SheetSyncPreset]]:
    grouped: dict[str, list[SheetSyncPreset]] = {}
    for preset in presets:
        grouped.setdefault(get_sheet_sync_spreadsheet_label(preset), []).append(preset)
    return grouped


def has_sheet_sync_preset_content(preset: SheetSyncPreset) -> bool:
    return any(
        (
            preset.spreadsheet_label.strip(),
            preset.name.strip(),
            preset.service_account_file.strip(),
            preset.spreadsheet_id.strip(),
            preset.sheet_name.strip(),
        )
    )


def get_sheet_sync_save_status(
    current_preset: SheetSyncPreset,
    *,
    selected_name: str,
    saved_preset: SheetSyncPreset | None,
) -> tuple[str, str]:
    if saved_preset is not None:
        if current_preset == saved_preset:
            return "success", f"保存済み: {get_sheet_sync_spreadsheet_label(saved_preset)} / {saved_preset.name}"
        return "warning", "未保存の変更があります。「設定を保存」で反映してください。"
    if has_sheet_sync_preset_content(current_preset):
        return "warning", "この新規設定はまだ保存されていません。"
    if selected_name == "新規設定":
        return "info", "新規設定を編集中です。"
    return "info", "設定を確認してください。"


def build_sheet_sync_command(
    python_executable: str,
    preset: SheetSyncPreset,
    request: SheetSyncRunRequest,
) -> list[str]:
    effective_ocr_backend = resolve_sheet_sync_ocr_backend(request)
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
        "--ocr-backend",
        effective_ocr_backend,
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
    command.append("--auto-detect-layout")

    if request.mode == RUN_MODE_TEST_ONE:
        command.extend(["--limit", "1"])
    elif request.mode == RUN_MODE_RETRY_ERRORS:
        command.append("--retry-errors")
    elif request.mode == RUN_MODE_RERUN_IMAGES:
        command.extend(["--only-style", preset.image_style, "--reprocess-existing"])
    elif request.mode == RUN_MODE_RERUN_AUDIO:
        command.extend(["--only-style", preset.audio_style, "--reprocess-existing"])
    elif request.mode == RUN_MODE_RERUN_LOW_QUALITY_OCR:
        command.extend(["--only-style", preset.image_style, "--rerun-low-quality-ocr"])

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


def _normalize_sheet_sync_preset(preset: SheetSyncPreset) -> SheetSyncPreset:
    return SheetSyncPreset(
        name=preset.name,
        service_account_file=preset.service_account_file,
        spreadsheet_id=preset.spreadsheet_id,
        sheet_name=preset.sheet_name,
        url_column=preset.url_column,
        style_column=preset.style_column,
        transcription_column=preset.transcription_column,
        status_column=preset.status_column,
        processed_at_column=preset.processed_at_column,
        error_column=preset.error_column,
        audio_style=preset.audio_style,
        image_style=preset.image_style,
        google_python=preset.google_python,
        spreadsheet_label=get_sheet_sync_spreadsheet_label(preset),
    )


def _coerce_preset(payload: dict[str, str]) -> SheetSyncPreset:
    default = default_sheet_sync_preset()
    values = asdict(default)
    for key in values:
        if key in payload:
            values[key] = str(payload[key])
    return _normalize_sheet_sync_preset(SheetSyncPreset(**values))
