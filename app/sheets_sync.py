from __future__ import annotations

import argparse
import json
import os
import site
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .downloader import DownloadError
from .models import TranscriptionResult
from .utils import ensure_directory

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
PROCESSING_STATUS = "処理中"
DONE_STATUS = "完了"
DONE_EMPTY_STATUS = "完了(空結果)"
ERROR_STATUS = "エラー"
PENDING_STATUSES = {"", "未処理", "再実行"}


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

    @property
    def supported_styles(self) -> set[str]:
        return {self.audio_style, self.image_style}


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

    context = ProcessingContext(
        output_dir=output_dir,
        download_dir=download_dir,
        cookies_file=cookies_file,
        audio_model_size=args.audio_model_size,
        audio_language=None if args.audio_language == "auto" else args.audio_language,
        ocr_languages=langs,
        ocr_min_confidence=args.min_ocr_confidence,
        ocr_gpu=args.gpu,
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
    values = fetch_sheet_values(sheets=sheets, config=config)
    pending_rows = list(
        iter_pending_rows(
            values=values,
            config=config,
            retry_errors=args.retry_errors,
            reprocess_existing=args.reprocess_existing,
            only_style=args.only_style.strip(),
        )
    )
    if args.limit > 0:
        pending_rows = pending_rows[: args.limit]

    if not pending_rows:
        print("[INFO] No pending rows found.")
        return 0

    failures = 0
    for row in pending_rows:
        print(f"[INFO] row={row.row_number} style={row.style} input={row.input_ref}")
        write_row_result(
            sheets=sheets,
            config=config,
            row_number=row.row_number,
            transcription=row.transcription,
            status=PROCESSING_STATUS,
            processed_at="",
            error="",
        )
        try:
            result = process_sheet_row(row=row, context=context, config=config)
            full_text = result.to_full_text()
            status = DONE_STATUS if full_text else DONE_EMPTY_STATUS
            write_row_result(
                sheets=sheets,
                config=config,
                row_number=row.row_number,
                transcription=full_text,
                status=status,
                processed_at=_timestamp_now(),
                error="",
            )
            print(
                f"[DONE] row={row.row_number} status={status} mode={result.transcription_mode} segments={len(result.segments)}"
            )
        except (FileNotFoundError, DownloadError, RuntimeError) as exc:
            failures += 1
            write_row_result(
                sheets=sheets,
                config=config,
                row_number=row.row_number,
                transcription=row.transcription,
                status=ERROR_STATUS,
                processed_at=_timestamp_now(),
                error=_truncate_cell_text(str(exc)),
            )
            print(f"[ERROR] row={row.row_number}: {exc}")

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


def write_row_result(
    sheets: SheetsBridgeContext,
    config: SheetSyncConfig,
    row_number: int,
    transcription: str,
    status: str,
    processed_at: str,
    error: str,
) -> None:
    data = [
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


def iter_pending_rows(
    values: list[list[str]],
    config: SheetSyncConfig,
    retry_errors: bool = False,
    reprocess_existing: bool = False,
    only_style: str = "",
) -> list[SheetRow]:
    rows: list[SheetRow] = []
    for row_number, raw_row in enumerate(values, start=config.start_row):
        row = SheetRow(
            row_number=row_number,
            input_ref=_get_cell(raw_row, config.url_column),
            style=_get_cell(raw_row, config.style_column),
            transcription=_get_cell(raw_row, config.transcription_column),
            status=_get_cell(raw_row, config.status_column),
            error=_get_cell(raw_row, config.error_column),
        )
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


def process_sheet_row(row: SheetRow, context: ProcessingContext, config: SheetSyncConfig) -> TranscriptionResult:
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
    )
    return result


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

    completed = subprocess.run(
        helper_args,
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=json.dumps(stdin_payload, ensure_ascii=False) if stdin_payload is not None else None,
    )
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
