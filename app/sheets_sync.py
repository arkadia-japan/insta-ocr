from __future__ import annotations

import argparse
import json
import os
import re
import site
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .downloader import DownloadError
from .models import TranscriptionResult
from .utils import detect_platform, ensure_directory, is_url, looks_like_image_url, normalize_text, safe_stem_from_input

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
PROCESSING_STATUS = "処理中"
DONE_STATUS = "完了"
DONE_EMPTY_STATUS = "完了(空結果)"
ERROR_STATUS = "エラー"
PENDING_STATUSES = {"", "未処理", "再実行"}
PREVIEW_MAX_COLUMN = "AZ"
PREVIEW_MAX_ROWS = 20
HEADER_SCAN_ROWS = 12
PROCESSING_COLUMN_FIELDS = (
    "style_column",
    "transcription_column",
    "status_column",
    "processed_at_column",
    "error_column",
)
PROCESSING_HEADER_LABELS = {
    "style_column": "投稿タイプ",
    "transcription_column": "文字起こし",
    "status_column": "処理ステータス",
    "processed_at_column": "処理日時",
    "error_column": "エラー内容",
}
HEADER_ALIASES = {
    "url_column": (
        "url",
        "動画url",
        "投稿url",
        "素材url",
        "動画リンク",
        "投稿リンク",
        "リンク",
        "link",
        "videourl",
        "reelurl",
        "shortsurl",
        "tiktokurl",
    ),
    "style_column": (
        "スタイル",
        "種別",
        "タイプ",
        "形式",
        "投稿形式",
        "動画形式",
        "投稿種別",
        "style",
        "mode",
        "投稿タイプ",
        "動画種別",
        "リール種別",
    ),
    "transcription_column": (
        "文字起こし",
        "書き起こし",
        "全文",
        "字幕",
        "台詞",
        "セリフ",
        "原稿",
        "台本",
        "transcription",
        "transcript",
    ),
    "status_column": (
        "状態",
        "ステータス",
        "処理状態",
        "進捗",
        "作業状況",
        "実施状況",
        "進行状況",
        "status",
        "処理ステータス",
    ),
    "processed_at_column": (
        "処理日時",
        "更新日時",
        "完了日時",
        "処理時刻",
        "processedat",
    ),
    "error_column": (
        "エラー",
        "エラー内容",
        "エラーメッセージ",
        "error",
    ),
}
AUDIO_STYLE_HINTS = ("音声", "asr", "voice", "talk", "ナレーション", "トーク", "会話")
IMAGE_STYLE_HINTS = ("画像", "ocr", "slide", "静止画", "写真", "図解", "テロップ", "caption", "text")
LOW_QUALITY_AVERAGE_CONFIDENCE_THRESHOLD = 0.82
LOW_QUALITY_SEGMENT_CONFIDENCE_THRESHOLD = 0.72
LOW_QUALITY_SEGMENT_RATIO_THRESHOLD = 0.45
LOW_QUALITY_SHORT_TEXT_LENGTH = 16
LOW_QUALITY_MIN_LINE_COUNT = 6
LOW_QUALITY_SHORT_LINE_RATIO_THRESHOLD = 0.45
LOW_QUALITY_REPEAT_RATIO_THRESHOLD = 0.40
GOOGLE_SHEETS_HELPER_TIMEOUT_SEC = 45.0


@dataclass(frozen=True)
class SheetSyncConfig:
    spreadsheet_id: str
    sheet_name: str
    url_column: str
    style_column: str = "K"
    transcription_column: str = "L"
    status_column: str = "M"
    processed_at_column: str = "N"
    error_column: str = "O"
    start_row: int = 2
    audio_style: str = "音声リール"
    image_style: str = "画像リール"
    default_style: str = ""

    @property
    def supported_styles(self) -> set[str]:
        return {style for style in {self.audio_style, self.image_style, self.default_style} if style}


@dataclass(frozen=True)
class SheetRow:
    row_number: int
    input_ref: str
    style: str
    transcription: str
    status: str
    error: str


@dataclass(frozen=True)
class SheetsBridgeContext:
    service_account_file: Path
    spreadsheet_id: str
    google_python: str


@dataclass
class ProcessingContext:
    output_dir: Path
    download_dir: Path
    cookies_file: Path | None
    audio_model_size: str
    audio_language: str | None
    ocr_languages: list[str]
    ocr_min_confidence: float
    ocr_gpu: bool
    ocr_backend: str
    sample_fps: float
    scene_threshold: float
    text_change_threshold: float
    min_interval_sec: float
    similarity_threshold: float
    keep_video: bool
    json_indent: int
    _audio_transcriber: Any = None
    _ocr_engine: Any = None

    def get_audio_transcriber(self):
        backends = _load_processing_backends()
        if self._audio_transcriber is None:
            self._audio_transcriber = backends["AudioTranscriber"](
                model_size=self.audio_model_size,
                language=self.audio_language,
            )
        return self._audio_transcriber

    def get_ocr_engine(self):
        backends = _load_processing_backends()
        if self._ocr_engine is None:
            self._ocr_engine = backends["OcrEngine"](
                languages=self.ocr_languages,
                gpu=self.ocr_gpu,
                min_confidence=self.ocr_min_confidence,
                backend=self.ocr_backend,
            )
        return self._ocr_engine


def build_parser() -> argparse.ArgumentParser:
    runtime_paths = _load_runtime_paths()
    default_output_dir = str(runtime_paths["get_runtime_output_dir"]() / "sheets_sync")
    default_download_dir = str(runtime_paths["get_runtime_download_dir"]() / "sheets_sync")

    parser = argparse.ArgumentParser(
        description="Sync a Google Sheet with the local ASR/OCR pipelines for Instagram Reels and short videos."
    )
    parser.add_argument("--service-account-file", required=True, help="Path to a Google service account JSON file.")
    parser.add_argument("--spreadsheet-id", required=True, help="Target spreadsheet ID.")
    parser.add_argument("--sheet-name", default="投稿データ260301", help="Target tab name. Default: 投稿データ260301")
    parser.add_argument("--url-column", required=True, help="Column letter that contains the video URL.")
    parser.add_argument("--style-column", default="K", help="Column letter for the style field. Default: K")
    parser.add_argument("--transcription-column", default="L", help="Column letter to write transcription text. Default: L")
    parser.add_argument("--status-column", default="M", help="Column letter to write processing status. Default: M")
    parser.add_argument("--processed-at-column", default="N", help="Column letter to write processed time. Default: N")
    parser.add_argument("--error-column", default="O", help="Column letter to write error text. Default: O")
    parser.add_argument("--start-row", type=int, default=2, help="First data row number. Default: 2")
    parser.add_argument("--audio-style", default="音声リール", help="Style value that selects ASR. Default: 音声リール")
    parser.add_argument("--image-style", default="画像リール", help="Style value that selects OCR. Default: 画像リール")
    parser.add_argument("--output-dir", default=default_output_dir, help=f"Directory for JSON/TXT/SRT output. Default: {default_output_dir}")
    parser.add_argument("--download-dir", default=default_download_dir, help=f"Directory for temporary video downloads. Default: {default_download_dir}")
    parser.add_argument("--cookies-file", default=None, help="Optional Netscape cookies.txt file for login-required videos.")
    parser.add_argument("--audio-model-size", default="small", choices=["tiny", "base", "small"], help="faster-whisper model size. Default: small")
    parser.add_argument("--audio-language", default="ja", choices=["auto", "ja", "en"], help="Audio language hint. Default: ja")
    parser.add_argument("--langs", default="ja,en", help="OCR languages, comma-separated. Default: ja,en")
    parser.add_argument(
        "--ocr-backend",
        default="auto",
        choices=("auto", "paddle", "easyocr"),
        help="OCR backend to use. Default: auto",
    )
    parser.add_argument("--min-ocr-confidence", type=float, default=0.15, help="Minimum OCR confidence. Default: 0.15")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for OCR if available.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Frames analyzed per second for OCR. Default: 4.0")
    parser.add_argument("--scene-threshold", type=float, default=0.22, help="Scene change sensitivity for OCR. Default: 0.22")
    parser.add_argument("--text-change-threshold", type=float, default=0.055, help="Text-region change sensitivity for OCR. Default: 0.055")
    parser.add_argument("--min-interval-sec", type=float, default=0.4, help="Minimum interval between kept keyframes. Default: 0.4")
    parser.add_argument("--similarity-threshold", type=float, default=0.86, help="Text similarity threshold for merging OCR segments. Default: 0.86")
    parser.add_argument("--keep-video", action="store_true", help="Keep downloaded videos.")
    parser.add_argument("--json-indent", type=int, default=2, help="JSON indentation width. Default: 2")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of rows to process. Default: 0 (all pending rows)")
    parser.add_argument("--retry-errors", action="store_true", help="Retry rows whose status is エラー.")
    parser.add_argument(
        "--auto-detect-layout",
        action="store_true",
        help="Infer column positions and style names from the sheet header before processing.",
    )
    parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Reprocess rows even if transcription/status are already populated, except rows currently 処理中.",
    )
    parser.add_argument(
        "--only-style",
        default="",
        help="Optional style filter, for example 画像リール or 音声リール.",
    )
    parser.add_argument(
        "--google-python",
        default=os.environ.get("SHEETS_SYNC_GOOGLE_PYTHON", "python"),
        help="Python executable used only for Google Sheets API access. Default: python",
    )
    parser.add_argument(
        "--rerun-low-quality-ocr",
        action="store_true",
        help="Re-run only completed low-quality image OCR rows with EasyOCR.",
    )
    parser.add_argument(
        "--extra-site-packages",
        default="",
        help="Optional extra site-packages directory to append for Google API dependencies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_optional_site_packages(args.extra_site_packages)

    config = SheetSyncConfig(
        spreadsheet_id=args.spreadsheet_id,
        sheet_name=args.sheet_name,
        url_column=_normalize_column_letter(args.url_column),
        style_column=_normalize_column_letter(args.style_column),
        transcription_column=_normalize_column_letter(args.transcription_column),
        status_column=_normalize_column_letter(args.status_column),
        processed_at_column=_normalize_column_letter(args.processed_at_column),
        error_column=_normalize_column_letter(args.error_column),
        start_row=args.start_row,
        audio_style=args.audio_style.strip(),
        image_style=args.image_style.strip(),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve()
    cookies_file = Path(args.cookies_file).expanduser().resolve() if args.cookies_file else None
    ensure_directory(output_dir)
    ensure_directory(download_dir)

    langs = [token.strip() for token in args.langs.split(",") if token.strip()]
    if not langs:
        parser.error("At least one OCR language is required via --langs.")

    effective_ocr_backend = "easyocr" if args.rerun_low_quality_ocr else args.ocr_backend
    if args.rerun_low_quality_ocr and effective_ocr_backend != args.ocr_backend:
        _log("[INFO] low-quality rerun forces OCR backend=easyocr")

    context = ProcessingContext(
        output_dir=output_dir,
        download_dir=download_dir,
        cookies_file=cookies_file,
        audio_model_size=args.audio_model_size,
        audio_language=None if args.audio_language == "auto" else args.audio_language,
        ocr_languages=langs,
        ocr_min_confidence=args.min_ocr_confidence,
        ocr_gpu=args.gpu,
        ocr_backend=effective_ocr_backend,
        sample_fps=args.sample_fps,
        scene_threshold=args.scene_threshold,
        text_change_threshold=args.text_change_threshold,
        min_interval_sec=args.min_interval_sec,
        similarity_threshold=args.similarity_threshold,
        keep_video=args.keep_video,
        json_indent=args.json_indent,
    )

    sheets = SheetsBridgeContext(
        service_account_file=Path(args.service_account_file).expanduser().resolve(),
        spreadsheet_id=args.spreadsheet_id,
        google_python=args.google_python,
    )
    preview_rows: list[list[str]] = []
    if args.auto_detect_layout:
        preview_rows = fetch_sheet_preview_rows(sheets=sheets, sheet_name=config.sheet_name)
        config, detection_messages = infer_sheet_sync_config_from_preview(config=config, preview_rows=preview_rows)
        for message in detection_messages:
            _log(f"[INFO] auto-detect: {message}")
        for message in ensure_sheet_sync_headers(sheets=sheets, config=config, preview_rows=preview_rows):
            _log(f"[INFO] auto-detect: {message}")
    values = fetch_sheet_values(sheets=sheets, config=config)
    pending_rows = list(
        iter_pending_rows(
            values=values,
            config=config,
            retry_errors=args.retry_errors,
            reprocess_existing=args.reprocess_existing,
            only_style=args.only_style.strip(),
            rerun_low_quality_ocr=args.rerun_low_quality_ocr,
            quality_output_dir=output_dir,
        )
    )
    if args.limit > 0:
        pending_rows = pending_rows[: args.limit]
    if args.rerun_low_quality_ocr:
        for row in pending_rows:
            _, reason = assess_low_quality_sheet_row(
                row=row,
                config=config,
                output_dir=output_dir,
                only_style=args.only_style.strip(),
            )
            _log(f"[INFO] low-quality: row={row.row_number} {reason}")

    if not pending_rows:
        if args.auto_detect_layout:
            preview_rows = fetch_sheet_preview_rows(sheets=sheets, sheet_name=config.sheet_name)
            for message in describe_sheet_sync_preview(config=config, preview_rows=preview_rows):
                _log(f"[INFO] diagnose: {message}")
        _log("[INFO] No pending rows found.")
        return 0

    failures = 0
    for row in pending_rows:
        _log(f"[INFO] row={row.row_number} style={row.style} input={row.input_ref}")
        write_row_result(
            sheets=sheets,
            config=config,
            row_number=row.row_number,
            style=row.style,
            transcription=row.transcription,
            status=PROCESSING_STATUS,
            processed_at="",
            error="",
        )
        active_row = row
        try:
            result = process_sheet_row(
                row=active_row,
                context=context,
                config=config,
                status_callback=_build_row_status_reporter(active_row.row_number),
            )
            retry_row = _build_empty_audio_retry_row(row=active_row, config=config, result=result)
            if retry_row is not None:
                active_row = retry_row
                _log(
                    f"[INFO] row={active_row.row_number} empty audio result; "
                    f"switching style to {active_row.style} and retrying"
                )
                write_row_result(
                    sheets=sheets,
                    config=config,
                    row_number=active_row.row_number,
                    style=active_row.style,
                    transcription=row.transcription,
                    status=PROCESSING_STATUS,
                    processed_at="",
                    error="",
                )
                result = process_sheet_row(
                    row=active_row,
                    context=context,
                    config=config,
                    status_callback=_build_row_status_reporter(active_row.row_number),
                )
            full_text = result.to_full_text()
            status = DONE_STATUS if full_text else DONE_EMPTY_STATUS
            write_row_result(
                sheets=sheets,
                config=config,
                row_number=active_row.row_number,
                style=active_row.style,
                transcription=full_text,
                status=status,
                processed_at=_timestamp_now(),
                error="",
            )
            _log(
                f"[DONE] row={active_row.row_number} status={status} "
                f"mode={result.transcription_mode} segments={len(result.segments)}"
            )
        except (FileNotFoundError, DownloadError, RuntimeError) as exc:
            retry_row = _build_audio_error_retry_row(row=active_row, config=config, exc=exc)
            if retry_row is not None:
                active_row = retry_row
                _log(
                    f"[INFO] row={active_row.row_number} audio unavailable; "
                    f"switching style to {active_row.style} and retrying"
                )
                write_row_result(
                    sheets=sheets,
                    config=config,
                    row_number=active_row.row_number,
                    style=active_row.style,
                    transcription=row.transcription,
                    status=PROCESSING_STATUS,
                    processed_at="",
                    error="",
                )
                try:
                    result = process_sheet_row(
                        row=active_row,
                        context=context,
                        config=config,
                        status_callback=_build_row_status_reporter(active_row.row_number),
                    )
                    full_text = result.to_full_text()
                    status = DONE_STATUS if full_text else DONE_EMPTY_STATUS
                    write_row_result(
                        sheets=sheets,
                        config=config,
                        row_number=active_row.row_number,
                        style=active_row.style,
                        transcription=full_text,
                        status=status,
                        processed_at=_timestamp_now(),
                        error="",
                    )
                    _log(
                        f"[DONE] row={active_row.row_number} status={status} "
                        f"mode={result.transcription_mode} segments={len(result.segments)}"
                    )
                    continue
                except (FileNotFoundError, DownloadError, RuntimeError) as retry_exc:
                    exc = retry_exc
            failures += 1
            write_row_result(
                sheets=sheets,
                config=config,
                row_number=active_row.row_number,
                style=active_row.style,
                transcription=active_row.transcription,
                status=ERROR_STATUS,
                processed_at=_timestamp_now(),
                error=_truncate_cell_text(str(exc)),
            )
            _log(f"[ERROR] row={active_row.row_number}: {exc}")

    return 1 if failures else 0


def build_sheets_service(service_account_file: Path, extra_site_packages: str = ""):
    _configure_optional_site_packages(extra_site_packages)
    return SheetsBridgeContext(
        service_account_file=service_account_file,
        spreadsheet_id="",
        google_python=os.environ.get("SHEETS_SYNC_GOOGLE_PYTHON", "python"),
    )


def fetch_sheet_values(sheets: SheetsBridgeContext, config: SheetSyncConfig) -> list[list[str]]:
    payload = _run_google_sheets_helper(
        sheets=sheets,
        command="fetch-values",
        extra_args=["--range", build_fetch_range(config)],
    )
    return payload.get("values", [])


def fetch_sheet_preview_rows(
    sheets: SheetsBridgeContext,
    sheet_name: str,
    *,
    max_column: str = PREVIEW_MAX_COLUMN,
    max_rows: int = PREVIEW_MAX_ROWS,
) -> list[list[str]]:
    payload = _run_google_sheets_helper(
        sheets=sheets,
        command="fetch-values",
        extra_args=["--range", f"{_quote_sheet_name(sheet_name)}!A1:{max_column}{max_rows}"],
    )
    return payload.get("values", [])


def auto_detect_sheet_sync_config(
    *,
    sheets: SheetsBridgeContext,
    config: SheetSyncConfig,
) -> tuple[SheetSyncConfig, list[str]]:
    preview_rows = fetch_sheet_preview_rows(sheets=sheets, sheet_name=config.sheet_name)
    return infer_sheet_sync_config_from_preview(config=config, preview_rows=preview_rows)


def infer_sheet_sync_config_from_preview(
    *,
    config: SheetSyncConfig,
    preview_rows: list[list[str]],
) -> tuple[SheetSyncConfig, list[str]]:
    if not preview_rows:
        return config, []

    detected_config = config
    messages: list[str] = []
    header_row_index, header_mapping = _detect_header_mapping(preview_rows)

    if header_row_index is not None:
        detected_start_row = header_row_index + 2
        if detected_start_row != detected_config.start_row:
            detected_config = replace(detected_config, start_row=detected_start_row)
        for field_name, column_letter in header_mapping.items():
            if getattr(detected_config, field_name) != column_letter:
                detected_config = replace(detected_config, **{field_name: column_letter})
    sample_mapping = _detect_sample_mapping(preview_rows, header_row_index)
    sample_updates: dict[str, str] = {}
    for field_name in ("url_column", "style_column", "status_column"):
        if field_name in header_mapping:
            continue
        candidate = sample_mapping.get(field_name, "")
        if candidate and getattr(detected_config, field_name) != candidate:
            sample_updates[field_name] = candidate
    if sample_updates:
        detected_config = replace(detected_config, **sample_updates)
        messages.append(
            "sample columns "
            + ", ".join(f"{field_name.removesuffix('_column')}={column}" for field_name, column in sample_updates.items())
        )

    allocated_columns: dict[str, str] = {}
    if header_row_index is not None:
        detected_config, allocated_columns = _allocate_processing_columns(
            config=detected_config,
            preview_rows=preview_rows,
            header_row_index=header_row_index,
            header_mapping=header_mapping,
            sample_updates=sample_updates,
        )
        if allocated_columns:
            messages.append(
                "allocated columns "
                + ", ".join(f"{field_name.removesuffix('_column')}={column}" for field_name, column in allocated_columns.items())
            )

    inferred_audio_style, inferred_image_style = _infer_style_values(
        preview_rows=preview_rows,
        style_column=detected_config.style_column,
        header_row_index=header_row_index,
        current_audio_style=detected_config.audio_style,
        current_image_style=detected_config.image_style,
    )
    if inferred_audio_style != detected_config.audio_style or inferred_image_style != detected_config.image_style:
        detected_config = replace(
            detected_config,
            audio_style=inferred_audio_style,
            image_style=inferred_image_style,
        )
        messages.append(f"styles audio={inferred_audio_style}, image={inferred_image_style}")

    if not _has_existing_style_column(header_mapping, sample_updates):
        default_style = _infer_default_style(
            preview_rows=preview_rows,
            url_column=detected_config.url_column,
            header_row_index=header_row_index,
            audio_style=detected_config.audio_style,
        )
        if default_style and default_style != detected_config.default_style:
            detected_config = replace(detected_config, default_style=default_style)
            messages.append(f"default style={default_style}")

    messages.insert(
        0,
        "columns "
        + ", ".join(
            [
                f"url={detected_config.url_column}",
                f"style={detected_config.style_column}",
                f"transcription={detected_config.transcription_column}",
                f"status={detected_config.status_column}",
                f"processed_at={detected_config.processed_at_column}",
                f"error={detected_config.error_column}",
                f"start_row={detected_config.start_row}",
            ]
        ),
    )

    return detected_config, messages


def ensure_sheet_sync_headers(
    *,
    sheets: SheetsBridgeContext,
    config: SheetSyncConfig,
    preview_rows: list[list[str]],
) -> list[str]:
    header_row_index, _ = _detect_header_mapping(preview_rows)
    if header_row_index is None:
        return []

    header_row_number = header_row_index + 1
    header_values = preview_rows[header_row_index]
    data: list[dict[str, object]] = []
    added_fields: list[str] = []
    for field_name in PROCESSING_COLUMN_FIELDS:
        column_letter = getattr(config, field_name)
        column_index = column_letter_to_index(column_letter)
        header_value = str(header_values[column_index]).strip() if column_index < len(header_values) else ""
        if header_value:
            continue
        data.append(
            {
                "range": f"{_quote_sheet_name(config.sheet_name)}!{column_letter}{header_row_number}",
                "values": [[PROCESSING_HEADER_LABELS[field_name]]],
            }
        )
        added_fields.append(f"{field_name.removesuffix('_column')}={column_letter}")

    if not data:
        return []

    _run_google_sheets_helper(
        sheets=sheets,
        command="batch-update",
        stdin_payload={"valueInputOption": "RAW", "data": data},
    )
    return ["added headers " + ", ".join(added_fields)]


def write_row_result(
    sheets: SheetsBridgeContext,
    config: SheetSyncConfig,
    row_number: int,
    style: str,
    transcription: str,
    status: str,
    processed_at: str,
    error: str,
) -> None:
    data = [
        {"range": f"{_quote_sheet_name(config.sheet_name)}!{config.style_column}{row_number}", "values": [[style]]},
        {"range": f"{_quote_sheet_name(config.sheet_name)}!{config.transcription_column}{row_number}", "values": [[_truncate_cell_text(transcription)]]},
        {"range": f"{_quote_sheet_name(config.sheet_name)}!{config.status_column}{row_number}", "values": [[status]]},
        {"range": f"{_quote_sheet_name(config.sheet_name)}!{config.processed_at_column}{row_number}", "values": [[processed_at]]},
        {"range": f"{_quote_sheet_name(config.sheet_name)}!{config.error_column}{row_number}", "values": [[_truncate_cell_text(error)]]},
    ]
    _run_google_sheets_helper(
        sheets=sheets,
        command="batch-update",
        stdin_payload={"valueInputOption": "RAW", "data": data},
    )


def _allocate_processing_columns(
    *,
    config: SheetSyncConfig,
    preview_rows: list[list[str]],
    header_row_index: int,
    header_mapping: dict[str, str],
    sample_updates: dict[str, str],
) -> tuple[SheetSyncConfig, dict[str, str]]:
    header_values = preview_rows[header_row_index]
    used_columns = {
        getattr(config, field_name)
        for field_name in ("url_column", *PROCESSING_COLUMN_FIELDS)
        if getattr(config, field_name)
    }
    updates: dict[str, str] = {}
    next_column_index = _next_available_column_index(preview_rows, used_columns)

    for field_name in PROCESSING_COLUMN_FIELDS:
        if field_name in header_mapping or field_name in sample_updates:
            continue
        current_column = getattr(config, field_name)
        sibling_columns = (used_columns | set(updates.values())) - {current_column}
        if _is_safe_processing_column(
            header_values=header_values,
            field_name=field_name,
            column_letter=current_column,
            sibling_columns=sibling_columns,
        ):
            continue
        while column_index_to_letter(next_column_index) in used_columns or column_index_to_letter(next_column_index) in updates.values():
            next_column_index += 1
        updates[field_name] = column_index_to_letter(next_column_index)
        next_column_index += 1

    if not updates:
        return config, {}
    return replace(config, **updates), updates


def _is_safe_processing_column(
    *,
    header_values: list[str],
    field_name: str,
    column_letter: str,
    sibling_columns: set[str],
) -> bool:
    if not column_letter or column_letter in sibling_columns:
        return False

    column_index = column_letter_to_index(column_letter)
    if column_index >= len(header_values):
        return True

    header_value = str(header_values[column_index]).strip()
    if not header_value:
        return True
    return _match_header_field(header_value) == field_name


def _next_available_column_index(preview_rows: list[list[str]], reserved_columns: set[str]) -> int:
    max_index = max((column_letter_to_index(column) for column in reserved_columns if column), default=-1)
    for row_values in preview_rows:
        for index, value in enumerate(row_values):
            if str(value).strip():
                max_index = max(max_index, index)
    return max_index + 1


def _has_existing_style_column(
    header_mapping: dict[str, str],
    sample_updates: dict[str, str],
) -> bool:
    return "style_column" in header_mapping or "style_column" in sample_updates


def _infer_default_style(
    *,
    preview_rows: list[list[str]],
    url_column: str,
    header_row_index: int | None,
    audio_style: str,
) -> str:
    if not audio_style:
        return ""

    data_rows = preview_rows[(header_row_index + 1) if header_row_index is not None else 0 :]
    platforms = {
        detect_platform(url)
        for row_values in data_rows
        for url in [_get_cell(row_values, url_column)]
        if is_url(url)
    }
    platforms.discard("")
    if platforms and platforms <= {"tiktok", "youtube"}:
        return audio_style
    return ""


def _resolve_row_style(*, raw_style: str, input_ref: str, config: SheetSyncConfig) -> str:
    if raw_style:
        return raw_style
    if config.default_style and detect_platform(input_ref) in {"tiktok", "youtube"}:
        return config.default_style
    return ""


def iter_pending_rows(
    values: list[list[str]],
    config: SheetSyncConfig,
    retry_errors: bool = False,
    reprocess_existing: bool = False,
    only_style: str = "",
    rerun_low_quality_ocr: bool = False,
    quality_output_dir: Path | None = None,
) -> list[SheetRow]:
    rows: list[SheetRow] = []
    for row_number, raw_row in enumerate(values, start=config.start_row):
        resolved_style = _resolve_row_style(
            raw_style=_get_cell(raw_row, config.style_column),
            input_ref=_get_cell(raw_row, config.url_column),
            config=config,
        )
        row = SheetRow(
            row_number=row_number,
            input_ref=_get_cell(raw_row, config.url_column),
            style=resolved_style,
            transcription=_get_cell(raw_row, config.transcription_column),
            status=_get_cell(raw_row, config.status_column),
            error=_get_cell(raw_row, config.error_column),
        )
        retry_done_empty_row = _build_done_empty_image_retry_row(row=row, config=config)
        candidate_row = retry_done_empty_row or row
        if rerun_low_quality_ocr:
            if quality_output_dir is None:
                continue
            should_rerun, _ = assess_low_quality_sheet_row(
                row=candidate_row,
                config=config,
                output_dir=quality_output_dir,
                only_style=only_style,
            )
            if should_rerun:
                rows.append(candidate_row)
            continue
        if retry_done_empty_row is not None:
            if not only_style or retry_done_empty_row.style == only_style:
                rows.append(retry_done_empty_row)
            continue
        if should_process_row(
            row=row,
            config=config,
            retry_errors=retry_errors,
            reprocess_existing=reprocess_existing,
            only_style=only_style,
        ):
            rows.append(row)
    return rows


def should_process_row(
    row: SheetRow,
    config: SheetSyncConfig,
    retry_errors: bool = False,
    reprocess_existing: bool = False,
    only_style: str = "",
) -> bool:
    if not row.input_ref or row.style not in config.supported_styles:
        return False
    if only_style and row.style != only_style:
        return False
    if row.status == PROCESSING_STATUS:
        return False
    if reprocess_existing:
        return True
    if row.transcription.strip():
        return False
    if row.status in {DONE_STATUS, DONE_EMPTY_STATUS}:
        return False
    if row.status in PENDING_STATUSES:
        return True
    if retry_errors and row.status.startswith(ERROR_STATUS):
        return True
    return False


def assess_low_quality_sheet_row(
    *,
    row: SheetRow,
    config: SheetSyncConfig,
    output_dir: Path,
    only_style: str = "",
) -> tuple[bool, str]:
    if not row.input_ref or row.style not in config.supported_styles:
        return False, "unsupported row"
    if row.style != config.image_style:
        return False, "style is not image"
    if only_style and row.style != only_style:
        return False, "style filter mismatch"
    if row.status == PROCESSING_STATUS:
        return False, "already processing"
    if row.status == DONE_EMPTY_STATUS:
        return True, "status is done-empty"
    if row.status != DONE_STATUS:
        return False, f"status={row.status or 'blank'}"

    current_text = normalize_text(row.transcription)
    if not current_text:
        return True, "sheet transcription is empty"

    output_payload = _load_existing_sheet_output(row=row, output_dir=output_dir)
    if output_payload is None:
        if _looks_like_low_quality_text(row.transcription):
            return True, "missing JSON and transcription looks weak"
        return False, "missing JSON but transcription looks usable"

    segments = output_payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return True, "output JSON has no segments"

    texts: list[str] = []
    confidences: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_text = normalize_text(str(segment.get("text", "")))
        if not segment_text:
            continue
        texts.append(segment_text)
        confidence = segment.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    if not texts:
        return True, "output JSON has no non-empty segments"

    if confidences:
        average_confidence = sum(confidences) / len(confidences)
        low_confidence_ratio = (
            sum(1 for value in confidences if value < LOW_QUALITY_SEGMENT_CONFIDENCE_THRESHOLD) / len(confidences)
        )
        if average_confidence < LOW_QUALITY_AVERAGE_CONFIDENCE_THRESHOLD:
            return True, f"avg_conf={average_confidence:.3f}"
        if len(confidences) >= 2 and low_confidence_ratio >= LOW_QUALITY_SEGMENT_RATIO_THRESHOLD:
            return True, f"low_conf_ratio={low_confidence_ratio:.2f}"

    combined_text = "\n".join(texts)
    if _looks_like_low_quality_text(combined_text):
        return True, "transcription looks noisy"
    return False, "quality looks acceptable"


def _load_existing_sheet_output(*, row: SheetRow, output_dir: Path) -> dict[str, Any] | None:
    output_path = output_dir / f"{safe_stem_from_input(row.input_ref)}.json"
    if not output_path.exists():
        return None
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _looks_like_low_quality_text(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) <= LOW_QUALITY_SHORT_TEXT_LENGTH:
        return True
    lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
    if len(lines) < LOW_QUALITY_MIN_LINE_COUNT:
        return False
    short_line_ratio = sum(1 for line in lines if len(line) <= 3) / len(lines)
    repeat_ratio = 1.0 - (len(set(lines)) / len(lines))
    return (
        short_line_ratio >= LOW_QUALITY_SHORT_LINE_RATIO_THRESHOLD
        or repeat_ratio >= LOW_QUALITY_REPEAT_RATIO_THRESHOLD
    )


def process_sheet_row(
    row: SheetRow,
    context: ProcessingContext,
    config: SheetSyncConfig,
    status_callback=None,
) -> TranscriptionResult:
    backends = _load_processing_backends()
    if row.style == config.audio_style:
        result, _ = backends["run_single_audio_input"](
            input_ref=row.input_ref,
            audio_transcriber=context.get_audio_transcriber(),
            options=backends["AudioProcessingOptions"](
                keep_video=context.keep_video,
                cookies_file=context.cookies_file,
                runtime_video_dir=context.download_dir,
                json_indent=context.json_indent,
            ),
            output_dir=context.output_dir,
            download_dir=context.download_dir,
            status_callback=status_callback,
        )
        return result

    result, _ = backends["run_single_input"](
        input_ref=row.input_ref,
        ocr_engine=context.get_ocr_engine(),
        options=backends["ProcessingOptions"](
            sample_fps=context.sample_fps,
            scene_threshold=context.scene_threshold,
            text_change_threshold=context.text_change_threshold,
            min_interval_sec=context.min_interval_sec,
            similarity_threshold=context.similarity_threshold,
            json_indent=context.json_indent,
            keep_video=context.keep_video,
            cookies_file=context.cookies_file,
            runtime_video_dir=context.download_dir,
        ),
        output_dir=context.output_dir,
        download_dir=context.download_dir,
        status_callback=status_callback,
    )
    return result


def _log(message: str) -> None:
    print(message, flush=True)


def _build_row_status_reporter(row_number: int):
    def report(message: str) -> None:
        _log(f"[INFO] row={row_number} {message}")

    return report


def _build_done_empty_image_retry_row(
    *,
    row: SheetRow,
    config: SheetSyncConfig,
) -> SheetRow | None:
    if row.style != config.audio_style:
        return None
    if not config.image_style or config.image_style == row.style:
        return None
    if row.transcription.strip():
        return None
    if row.status != DONE_EMPTY_STATUS:
        return None
    return SheetRow(
        row_number=row.row_number,
        input_ref=row.input_ref,
        style=config.image_style,
        transcription="",
        status=row.status,
        error=row.error,
    )


def _build_empty_audio_retry_row(
    *,
    row: SheetRow,
    config: SheetSyncConfig,
    result: TranscriptionResult,
) -> SheetRow | None:
    if row.style != config.audio_style:
        return None
    if not config.image_style or config.image_style == row.style:
        return None
    if result.to_full_text().strip():
        return None
    return SheetRow(
        row_number=row.row_number,
        input_ref=row.input_ref,
        style=config.image_style,
        transcription=row.transcription,
        status=row.status,
        error=row.error,
    )


def _build_audio_error_retry_row(
    *,
    row: SheetRow,
    config: SheetSyncConfig,
    exc: Exception,
) -> SheetRow | None:
    if row.style != config.audio_style:
        return None
    if not config.image_style or config.image_style == row.style:
        return None
    if "no audio stream" not in str(exc).lower():
        return None
    return SheetRow(
        row_number=row.row_number,
        input_ref=row.input_ref,
        style=config.image_style,
        transcription=row.transcription,
        status=row.status,
        error=row.error,
    )


def _detect_header_mapping(preview_rows: list[list[str]]) -> tuple[int | None, dict[str, str]]:
    best_index: int | None = None
    best_mapping: dict[str, str] = {}
    best_score = -1

    for row_index, row_values in enumerate(preview_rows[:HEADER_SCAN_ROWS]):
        mapping: dict[str, str] = {}
        for column_index, cell_value in enumerate(row_values):
            field_name = _match_header_field(cell_value)
            if not field_name or field_name in mapping:
                continue
            mapping[field_name] = column_index_to_letter(column_index)

        score = len(mapping) * 100 + _header_row_signal(row_values)
        if score > best_score:
            best_index = row_index
            best_mapping = mapping
            best_score = score

    if best_index is None:
        return None, {}
    if len(best_mapping) >= 2:
        return best_index, best_mapping
    if len(best_mapping) >= 1 and _header_row_signal(preview_rows[best_index]) >= 4:
        return best_index, best_mapping
    if _header_row_signal(preview_rows[best_index]) >= 6:
        return best_index, best_mapping
    return None, {}


def _header_row_signal(row_values: list[str]) -> int:
    score = 0
    for cell_value in row_values:
        text = str(cell_value or "").strip()
        if not text:
            continue
        if is_url(text):
            continue
        normalized = _normalize_header_text(text)
        if not normalized:
            continue
        if len(normalized) <= 20:
            score += 1
        if any(token in normalized for token in ("url", "id", "写真", "画像", "投稿", "状態", "文字起こし", "ユーザー", "説明", "リンク")):
            score += 1
    return score


def _match_header_field(value: str) -> str:
    normalized = _normalize_header_text(value)
    if not normalized:
        return ""
    if any(
        token in normalized
        for token in ("サムネ", "thumbnail", "プロフ", "プロフィール", "avatar", "icon", "アイコン")
    ):
        if any(token in normalized for token in ("url", "リンク", "link")):
            return ""

    best_field = ""
    best_score = -1
    for field_name, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_header_text(alias)
            if not normalized_alias:
                continue
            if normalized == normalized_alias:
                score = len(normalized_alias) + 100
            elif normalized_alias in normalized:
                score = len(normalized_alias)
            else:
                continue
            if score > best_score:
                best_score = score
                best_field = field_name
    return best_field


def _detect_sample_mapping(preview_rows: list[list[str]], header_row_index: int | None) -> dict[str, str]:
    data_rows = preview_rows[(header_row_index + 1) if header_row_index is not None else 0 :]
    if not data_rows:
        return {}

    column_count = max((len(row_values) for row_values in data_rows), default=0)
    mapping: dict[str, str] = {}

    url_candidate = _best_column_by_score(data_rows, column_count, _score_url_like_cell)
    if url_candidate:
        mapping["url_column"] = url_candidate

    style_candidate = _best_column_by_score(data_rows, column_count, _score_style_like_cell)
    if style_candidate:
        mapping["style_column"] = style_candidate

    status_candidate = _best_column_by_score(data_rows, column_count, _score_status_like_cell)
    if status_candidate:
        mapping["status_column"] = status_candidate

    return mapping


def _best_column_by_score(
    rows: list[list[str]],
    column_count: int,
    scorer,
) -> str:
    best_index = -1
    best_score = 0
    second_score = 0

    for column_index in range(column_count):
        score = 0
        for row_values in rows:
            if column_index >= len(row_values):
                continue
            score += scorer(str(row_values[column_index]).strip())
        if score > best_score:
            second_score = best_score
            best_index = column_index
            best_score = score
        elif score > second_score:
            second_score = score

    if best_index < 0 or best_score <= 0:
        return ""
    if best_score == second_score:
        return ""
    return column_index_to_letter(best_index)


def _rank_columns_by_score(
    rows: list[list[str]],
    column_count: int,
    scorer,
    *,
    limit: int = 3,
) -> list[tuple[str, int]]:
    ranked: list[tuple[str, int]] = []
    for column_index in range(column_count):
        score = 0
        for row_values in rows:
            if column_index >= len(row_values):
                continue
            score += scorer(str(row_values[column_index]).strip())
        if score > 0:
            ranked.append((column_index_to_letter(column_index), score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def _format_ranked_columns_message(
    label: str,
    ranked_columns: list[tuple[str, int]],
    rows: list[list[str]],
) -> str:
    parts: list[str] = []
    for column_letter, score in ranked_columns:
        samples = _unique_non_empty_values(rows, column_letter)[:3]
        suffix = f" ({', '.join(samples)})" if samples else ""
        parts.append(f"{column_letter}:{score}{suffix}")
    return f"{label} " + "; ".join(parts)


def _unique_non_empty_values(rows: list[list[str]], column: str, *, limit: int = 6) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row_values in rows:
        value = _get_cell(row_values, column)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) >= limit:
            break
    return values


def _score_url_like_cell(value: str) -> int:
    if not value or not is_url(value):
        return 0
    if looks_like_image_url(value):
        return 0

    lowered = value.lower()
    if "tiktok.com/" in lowered and "/video/" in lowered:
        return 8
    if "instagram.com/reel/" in lowered:
        return 8
    if "youtube.com/shorts/" in lowered:
        return 8
    if "youtube.com/watch" in lowered or "youtu.be/" in lowered:
        return 7

    platform = detect_platform(value)
    return 3 if platform in {"instagram", "tiktok", "youtube"} else 1


def _score_status_like_cell(value: str) -> int:
    if not value:
        return 1
    if value in PENDING_STATUSES or value in {PROCESSING_STATUS, DONE_STATUS, DONE_EMPTY_STATUS}:
        return 3
    if value.startswith(ERROR_STATUS):
        return 3
    return 0


def _score_style_like_cell(value: str) -> int:
    if not value:
        return 0
    if is_url(value):
        return 0
    normalized = _normalize_header_text(value)
    if any(token in normalized for token in ("サムネ", "thumbnail", "プロフ", "プロフィール", "avatar", "icon", "アイコン")):
        return 0
    if any(token in normalized for token in ("style", "mode", "種別", "スタイル", "タイプ")):
        return 2
    if _style_hint_score(value, AUDIO_STYLE_HINTS) > 0 or _style_hint_score(value, IMAGE_STYLE_HINTS) > 0:
        return 3
    return 0


def _infer_style_values(
    *,
    preview_rows: list[list[str]],
    style_column: str,
    header_row_index: int | None,
    current_audio_style: str,
    current_image_style: str,
) -> tuple[str, str]:
    data_rows = preview_rows[(header_row_index + 1) if header_row_index is not None else 0 :]
    seen_values: list[str] = []
    seen_set: set[str] = set()
    for row_values in data_rows:
        value = _get_cell(row_values, style_column)
        if not value or value in seen_set:
            continue
        seen_set.add(value)
        seen_values.append(value)

    if not seen_values:
        return current_audio_style, current_image_style

    audio_style = current_audio_style if current_audio_style in seen_set else _best_style_match(seen_values, AUDIO_STYLE_HINTS)
    image_style = current_image_style if current_image_style in seen_set else _best_style_match(seen_values, IMAGE_STYLE_HINTS)
    if audio_style and image_style and audio_style != image_style:
        return audio_style, image_style
    return current_audio_style, current_image_style


def describe_sheet_sync_preview(
    *,
    config: SheetSyncConfig,
    preview_rows: list[list[str]],
) -> list[str]:
    messages: list[str] = []
    if not preview_rows:
        return ["preview is empty"]

    header_row_index, header_mapping = _detect_header_mapping(preview_rows)
    if header_row_index is not None:
        header_values = preview_rows[header_row_index]
        messages.append(
            f"header row={header_row_index + 1} values="
            + ", ".join(
                f"{column_index_to_letter(index)}={str(value).strip()}"
                for index, value in enumerate(header_values)
                if str(value).strip()
            )
        )
        if header_mapping:
            messages.append(
                "header mapping "
                + ", ".join(f"{field_name.removesuffix('_column')}={column}" for field_name, column in header_mapping.items())
            )

    data_rows = preview_rows[(header_row_index + 1) if header_row_index is not None else 0 :]
    column_count = max((len(row_values) for row_values in data_rows), default=0)
    if column_count == 0:
        return messages or ["no data rows found"]

    url_candidates = _rank_columns_by_score(data_rows, column_count, _score_url_like_cell)
    style_candidates = _rank_columns_by_score(data_rows, column_count, _score_style_like_cell)
    status_candidates = _rank_columns_by_score(data_rows, column_count, _score_status_like_cell)

    if url_candidates:
        messages.append(_format_ranked_columns_message("url candidates", url_candidates, data_rows))
    if style_candidates:
        messages.append(_format_ranked_columns_message("style candidates", style_candidates, data_rows))
    if status_candidates:
        messages.append(_format_ranked_columns_message("status candidates", status_candidates, data_rows))

    configured_style_values = _unique_non_empty_values(data_rows, config.style_column)
    if configured_style_values:
        messages.append(
            f"configured style column {config.style_column} sample values={', '.join(configured_style_values[:5])}"
        )
    configured_status_values = _unique_non_empty_values(data_rows, config.status_column)
    if configured_status_values:
        messages.append(
            f"configured status column {config.status_column} sample values={', '.join(configured_status_values[:5])}"
        )

    return messages


def _best_style_match(values: list[str], hints: tuple[str, ...]) -> str:
    best_value = ""
    best_score = 0
    for value in values:
        score = _style_hint_score(value, hints)
        if score > best_score:
            best_value = value
            best_score = score
    return best_value


def _style_hint_score(value: str, hints: tuple[str, ...]) -> int:
    normalized = _normalize_header_text(value)
    score = 0
    for hint in hints:
        normalized_hint = _normalize_header_text(hint)
        if normalized_hint and normalized_hint in normalized:
            score = max(score, len(normalized_hint))
    return score


def build_fetch_range(config: SheetSyncConfig) -> str:
    max_index = max(
        column_letter_to_index(config.url_column),
        column_letter_to_index(config.style_column),
        column_letter_to_index(config.transcription_column),
        column_letter_to_index(config.status_column),
        column_letter_to_index(config.processed_at_column),
        column_letter_to_index(config.error_column),
    )
    max_column = column_index_to_letter(max_index)
    return f"{_quote_sheet_name(config.sheet_name)}!A{config.start_row}:{max_column}"


def column_letter_to_index(column: str) -> int:
    normalized = _normalize_column_letter(column)
    value = 0
    for char in normalized:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def column_index_to_letter(index: int) -> str:
    if index < 0:
        raise ValueError("Column index must be 0 or greater.")
    result: list[str] = []
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        result.append(chr(ord("A") + remainder))
    return "".join(reversed(result))


def _normalize_column_letter(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized or not normalized.isalpha():
        raise ValueError(f"Invalid column letter: {value!r}")
    return normalized


def _normalize_header_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    normalized = re.sub(r"[\s_/\\()（）［］\[\]{}【】・:：-]+", "", normalized)
    return normalized


def _get_cell(row_values: list[str], column: str) -> str:
    index = column_letter_to_index(column)
    if index >= len(row_values):
        return ""
    return str(row_values[index]).strip()


def _quote_sheet_name(sheet_name: str) -> str:
    return "'" + sheet_name.replace("'", "''") + "'"


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _truncate_cell_text(value: str, limit: int = 45000) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[: limit - 14].rstrip() + "\n\n[Truncated]"


def _run_google_sheets_helper(
    sheets: SheetsBridgeContext,
    command: str,
    extra_args: list[str] | None = None,
    stdin_payload: dict | None = None,
) -> dict:
    _log(f"[INFO] sheets api: {command}")
    helper_args = [
        sheets.google_python,
        "-m",
        "app.google_sheets_bridge",
        command,
        "--service-account-file",
        str(sheets.service_account_file),
        "--spreadsheet-id",
        sheets.spreadsheet_id,
    ]
    if extra_args:
        helper_args.extend(extra_args)

    try:
        completed = subprocess.run(
            helper_args,
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=json.dumps(stdin_payload, ensure_ascii=False) if stdin_payload is not None else None,
            timeout=float(os.environ.get("SHEETS_SYNC_GOOGLE_HELPER_TIMEOUT_SEC", GOOGLE_SHEETS_HELPER_TIMEOUT_SEC)),
        )
    except subprocess.TimeoutExpired as exc:
        timeout_sec = float(os.environ.get("SHEETS_SYNC_GOOGLE_HELPER_TIMEOUT_SEC", GOOGLE_SHEETS_HELPER_TIMEOUT_SEC))
        partial_output = (getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or getattr(exc, "output", None) or "")
        partial_text = str(partial_output).strip()
        message = f"Google Sheets helper timed out after {timeout_sec:.0f}s: {command}"
        if partial_text:
            message = f"{message}\n{partial_text}"
        raise RuntimeError(message) from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"Google Sheets helper failed: {command}"
        raise RuntimeError(message)

    stdout = (completed.stdout or "").strip()
    if not stdout:
        return {}
    return json.loads(stdout)


def _configure_optional_site_packages(extra_site_packages: str = "") -> None:
    configured = (extra_site_packages or os.environ.get("SHEETS_SYNC_EXTRA_SITE_PACKAGES", "")).strip()
    if not configured:
        return

    for raw_entry in configured.split(os.pathsep):
        entry = raw_entry.strip()
        if not entry:
            continue
        path = Path(entry).expanduser()
        if not path.exists():
            continue
        path_str = str(path)
        if path_str in sys.path:
            continue
        # Append Google API dependencies without shadowing venv-native packages such as numpy/cv2.
        site.addsitedir(path_str)


@lru_cache(maxsize=1)
def _load_runtime_paths() -> dict[str, Any]:
    from .runtime_paths import get_runtime_download_dir, get_runtime_output_dir

    return {
        "get_runtime_download_dir": get_runtime_download_dir,
        "get_runtime_output_dir": get_runtime_output_dir,
    }


@lru_cache(maxsize=1)
def _load_processing_backends() -> dict[str, Any]:
    from .runtime_paths import configure_paddle_runtime_env

    configure_paddle_runtime_env()

    from .audio_pipeline import AudioProcessingOptions, run_single_audio_input
    from .audio_transcriber import AudioTranscriber
    from .ocr_engine import OcrEngine
    from .pipeline import ProcessingOptions, run_single_input

    return {
        "AudioProcessingOptions": AudioProcessingOptions,
        "AudioTranscriber": AudioTranscriber,
        "OcrEngine": OcrEngine,
        "ProcessingOptions": ProcessingOptions,
        "run_single_audio_input": run_single_audio_input,
        "run_single_input": run_single_input,
    }


if __name__ == "__main__":
    raise SystemExit(main())
