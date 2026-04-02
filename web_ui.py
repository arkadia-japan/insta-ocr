from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.audio_pipeline import AudioProcessingOptions, run_single_audio_input
from app.audio_transcriber import AudioTranscriber
from app.downloader import DownloadError
from app.ocr_engine import OcrEngine
from app.pipeline import ProcessingOptions, run_single_input
from app.runtime_paths import (
    get_runtime_download_dir,
    get_runtime_logs_dir,
    get_runtime_output_dir,
    get_runtime_upload_dir,
    get_runtime_root,
)
from app.sheets_dashboard import (
    RUN_MODE_PENDING,
    RUN_MODE_RERUN_AUDIO,
    RUN_MODE_RERUN_IMAGES,
    RUN_MODE_RERUN_LOW_QUALITY_OCR,
    RUN_MODE_RETRY_ERRORS,
    RUN_MODE_TEST_ONE,
    SheetSyncPreset,
    SheetSyncRunRequest,
    build_sheet_sync_command,
    default_sheet_sync_preset,
    get_sheet_sync_save_status,
    get_sheet_sync_spreadsheet_label,
    get_sheet_dashboard_config_path,
    group_sheet_sync_presets,
    load_sheet_sync_presets,
    read_text_file_best_effort,
    resolve_sheet_sync_ocr_backend,
    resolve_sheet_sync_selected_preset,
    save_sheet_sync_presets,
    summarize_sheet_sync_output,
    upsert_sheet_sync_preset,
)
from app.utils import ensure_directory, format_compact_time

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = get_runtime_root()
DEFAULT_OUTPUT_DIR = get_runtime_output_dir()
DEFAULT_DOWNLOAD_DIR = get_runtime_download_dir()
DEFAULT_UPLOAD_DIR = get_runtime_upload_dir()
DEFAULT_LOG_DIR = get_runtime_logs_dir()
INSTAGRAM_REEL_PATTERN = re.compile(r"^https?://(www\.)?instagram\.com/reel/[^/\s?#]+/?(?:\?.*)?$", re.IGNORECASE)
SHEET_SYNC_LOG_PATH = DEFAULT_LOG_DIR / "sheet_sync_latest.log"
SHEET_SYNC_SELECTED_SPREADSHEET_WIDGET_KEY = "sheet_sync_selected_spreadsheet_widget"
SHEET_SYNC_PENDING_SELECTED_SPREADSHEET_KEY = "sheet_sync_pending_selected_spreadsheet"
SHEET_SYNC_SELECTED_PRESET_WIDGET_KEY = "sheet_sync_selected_preset_widget"
SHEET_SYNC_PENDING_SELECTED_PRESET_KEY = "sheet_sync_pending_selected_preset"
SHEET_SYNC_TAB_CHOICE_WIDGET_KEY = "sheet_sync_tab_choice_widget"
SHEET_SYNC_TAB_CACHE_KEY = "sheet_sync_tab_cache"
SHEET_SYNC_MANUAL_TAB_OPTION = "手入力"
OCR_BACKEND_LABELS = {
    "auto": "自動 (PaddleOCR -> EasyOCR)",
    "easyocr": "EasyOCR",
    "paddle": "PaddleOCR",
}
SHEET_SYNC_PRESET_OPTIONS = {
    RUN_MODE_PENDING: "未処理をまとめて実行",
    RUN_MODE_TEST_ONE: "未処理を1件だけテスト",
    RUN_MODE_RETRY_ERRORS: "エラー行だけ再実行",
    RUN_MODE_RERUN_IMAGES: "画像リールを再実行",
    RUN_MODE_RERUN_LOW_QUALITY_OCR: "低品質行だけEasyOCR再実行",
    RUN_MODE_RERUN_AUDIO: "音声リールを再実行",
}


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._-")
    return cleaned or "upload.mp4"


def _save_uploaded_files(uploaded_files, target_dir: Path) -> list[str]:
    ensure_directory(target_dir)
    paths: list[str] = []
    for uploaded in uploaded_files:
        safe_name = _safe_filename(uploaded.name)
        path = target_dir / f"upload_{uuid4().hex}_{safe_name}"
        path.write_bytes(uploaded.getbuffer())
        paths.append(str(path))
    return paths


def _is_instagram_reel_url(value: str) -> bool:
    return bool(INSTAGRAM_REEL_PATTERN.match(value.strip()))


@st.cache_resource(show_spinner=False)
def _build_ocr_engine(languages_key: str, gpu: bool, min_confidence: float, backend: str) -> OcrEngine:
    languages = [token.strip() for token in languages_key.split(",") if token.strip()]
    return OcrEngine(languages=languages, gpu=gpu, min_confidence=min_confidence, backend=backend)


@st.cache_resource(show_spinner=False)
def _build_audio_transcriber(model_size: str, language_key: str) -> AudioTranscriber:
    language = None if language_key == "auto" else language_key
    return AudioTranscriber(model_size=model_size, language=language)


def _make_status_reporter(status_box, details_box, input_ref: str, index: int, total: int):
    def report(message: str) -> None:
        status_box.update(label=f"{index}/{total} 件目を処理中", state="running")
        details_box.info(f"{input_ref}\n{message}")

    return report


def _build_processing_options(
    sample_fps: float,
    scene_threshold: float,
    text_change_threshold: float,
    min_interval_sec: float,
    similarity_threshold: float,
    keep_video: bool,
    cookies_file: Path | None,
) -> ProcessingOptions:
    return ProcessingOptions(
        sample_fps=sample_fps,
        scene_threshold=scene_threshold,
        text_change_threshold=text_change_threshold,
        min_interval_sec=min_interval_sec,
        similarity_threshold=similarity_threshold,
        keep_video=keep_video,
        cookies_file=cookies_file,
        runtime_video_dir=DEFAULT_DOWNLOAD_DIR,
    )


def _build_audio_processing_options(
    keep_video: bool,
    cookies_file: Path | None,
) -> AudioProcessingOptions:
    return AudioProcessingOptions(
        keep_video=keep_video,
        cookies_file=cookies_file,
        runtime_video_dir=DEFAULT_DOWNLOAD_DIR,
    )


def _run_transcription(
    session_prefix: str,
    inputs: list[str],
    ocr_engine: OcrEngine,
    options: ProcessingOptions,
    output_dir: Path,
    download_dir: Path,
    status_placeholder,
    details_placeholder,
    progress_placeholder,
) -> None:
    success_records: list[dict] = []
    failures: list[str] = []

    status_box = status_placeholder.status("処理を開始します", expanded=True)
    details_box = details_placeholder.empty()
    progress = progress_placeholder.progress(0.0)

    for index, input_ref in enumerate(inputs, start=1):
        reporter = _make_status_reporter(
            status_box=status_box,
            details_box=details_box,
            input_ref=input_ref,
            index=index,
            total=len(inputs),
        )
        try:
            result, paths = run_single_input(
                input_ref=input_ref,
                ocr_engine=ocr_engine,
                options=options,
                output_dir=output_dir,
                download_dir=download_dir,
                status_callback=reporter,
            )
            success_records.append(
                {
                    "result": result,
                    "paths": {name: str(path) for name, path in paths.items()},
                }
            )
        except (FileNotFoundError, DownloadError, RuntimeError) as exc:
            failures.append(f"{input_ref}: {exc}")
            details_box.error(f"{input_ref}\n{exc}")

        progress.progress(index / len(inputs))

    if failures:
        status_box.update(label="一部の入力でエラーが発生しました", state="error")
    else:
        status_box.update(label="文字起こしが完了しました", state="complete")

    st.session_state[f"{session_prefix}_success_records"] = success_records
    st.session_state[f"{session_prefix}_failures"] = failures
    st.session_state[f"{session_prefix}_output_dir"] = str(output_dir)


def _run_audio_transcription(
    session_prefix: str,
    inputs: list[str],
    audio_transcriber: AudioTranscriber,
    options: AudioProcessingOptions,
    output_dir: Path,
    download_dir: Path,
    status_placeholder,
    details_placeholder,
    progress_placeholder,
) -> None:
    success_records: list[dict] = []
    failures: list[str] = []

    status_box = status_placeholder.status("処理を開始します", expanded=True)
    details_box = details_placeholder.empty()
    progress = progress_placeholder.progress(0.0)

    for index, input_ref in enumerate(inputs, start=1):
        reporter = _make_status_reporter(
            status_box=status_box,
            details_box=details_box,
            input_ref=input_ref,
            index=index,
            total=len(inputs),
        )
        try:
            result, paths = run_single_audio_input(
                input_ref=input_ref,
                audio_transcriber=audio_transcriber,
                options=options,
                output_dir=output_dir,
                download_dir=download_dir,
                status_callback=reporter,
            )
            success_records.append(
                {
                    "result": result,
                    "paths": {name: str(path) for name, path in paths.items()},
                }
            )
        except (FileNotFoundError, DownloadError, RuntimeError) as exc:
            failures.append(f"{input_ref}: {exc}")
            details_box.error(f"{input_ref}\n{exc}")

        progress.progress(index / len(inputs))

    if failures:
        status_box.update(label="一部の入力でエラーが発生しました", state="error")
    else:
        status_box.update(label="文字起こしが完了しました", state="complete")

    st.session_state[f"{session_prefix}_success_records"] = success_records
    st.session_state[f"{session_prefix}_failures"] = failures
    st.session_state[f"{session_prefix}_output_dir"] = str(output_dir)



def _render_download_buttons(paths: dict[str, str], key_prefix: str) -> None:
    json_path = Path(paths["json"])
    txt_path = Path(paths["txt"])
    srt_path = Path(paths["srt"])

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "JSONをダウンロード",
        data=json_path.read_bytes(),
        file_name=json_path.name,
        mime="application/json",
        key=f"{key_prefix}_json",
    )
    col2.download_button(
        "TXTをダウンロード",
        data=txt_path.read_bytes(),
        file_name=txt_path.name,
        mime="text/plain",
        key=f"{key_prefix}_txt",
    )
    col3.download_button(
        "SRTをダウンロード",
        data=srt_path.read_bytes(),
        file_name=srt_path.name,
        mime="application/x-subrip",
        key=f"{key_prefix}_srt",
    )



def _render_result(record: dict, index: int, key_prefix: str, show_full_text_first: bool) -> None:
    result = record["result"]
    paths = record["paths"]
    mode = getattr(result, "transcription_mode", "ocr")

    st.subheader(f"{index}. {result.input_ref}")

    full_text = result.to_full_text()
    if show_full_text_first:
        st.text_area(
            "全文",
            value=full_text,
            height=340,
            key=f"{key_prefix}_full_text_{index}",
        )
        with st.expander("タイムスタンプ付き全文を見る"):
            st.text_area(
                "タイムスタンプ付き全文",
                value=result.to_full_text(include_timestamps=True),
                height=320,
                key=f"{key_prefix}_full_text_timestamped_{index}",
            )

    col1, col2, col3 = st.columns(3)
    col1.metric("出力セグメント数", len(result.segments))
    if mode == "asr":
        col2.metric("認識方式", "音声")
    else:
        col2.metric("解析セグメント数", result.sampled_frames)
    col3.metric("動画長", format_compact_time(result.duration_sec))
    if mode == "asr":
        st.caption(f"音声セグメント数: {len(result.segments)}")
    else:
        st.caption(f"OCRヒット数: {result.ocr_hits} / fallback使用: {result.fallback_used}")

    if not result.segments:
        if mode == "asr":
            st.warning("音声が認識できませんでした。動画に音声があるか、URLへアクセスできるか確認して再実行してください。")
        else:
            st.warning(
                "テキストが抽出できませんでした。解析FPS、文字変化しきい値、OCR最低信頼度を調整して再実行してください。"
            )

    if not show_full_text_first:
        with st.expander("全文を見る", expanded=True):
            st.text_area(
                "全文",
                value=full_text,
                height=300,
                key=f"{key_prefix}_general_full_text_{index}",
            )

    _render_download_buttons(paths=paths, key_prefix=f"{key_prefix}_download_{index}")

    segment_rows = [
        {
            "開始": format_compact_time(segment.start_sec),
            "終了": format_compact_time(segment.end_sec),
            "テキスト": segment.text,
            "信頼度": segment.confidence,
        }
        for segment in result.segments
    ]

    with st.expander("セグメント結果を見る"):
        st.dataframe(segment_rows, use_container_width=True)

    with st.expander("JSONを見る"):
        st.code(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), language="json")



def _render_saved_results(session_prefix: str, show_full_text_first: bool) -> None:
    success_records = st.session_state.get(f"{session_prefix}_success_records", [])
    failures = st.session_state.get(f"{session_prefix}_failures", [])
    output_dir_state = st.session_state.get(f"{session_prefix}_output_dir")

    if output_dir_state:
        st.info(f"出力先: {output_dir_state}")

    if failures:
        st.error("一部の入力で失敗しました。")
        for message in failures:
            st.write(f"- {message}")

    if not success_records:
        return

    st.success(f"{len(success_records)} 件の文字起こしが完了しました。")
    for index, record in enumerate(success_records, start=1):
        _render_result(
            record=record,
            index=index,
            key_prefix=session_prefix,
            show_full_text_first=show_full_text_first,
        )


def _new_sheet_sync_preset(
    *,
    base_preset: SheetSyncPreset | None = None,
    spreadsheet_label: str = "",
) -> SheetSyncPreset:
    base = base_preset or default_sheet_sync_preset()
    inherit_existing_sheet = base_preset is not None and bool(spreadsheet_label.strip())
    return SheetSyncPreset(
        name="",
        spreadsheet_label=spreadsheet_label.strip(),
        service_account_file=base.service_account_file,
        spreadsheet_id=base.spreadsheet_id if inherit_existing_sheet else "",
        sheet_name=base.sheet_name if inherit_existing_sheet else "",
        url_column=base.url_column,
        style_column=base.style_column,
        transcription_column=base.transcription_column,
        status_column=base.status_column,
        processed_at_column=base.processed_at_column,
        error_column=base.error_column,
        audio_style=base.audio_style,
        image_style=base.image_style,
        google_python=base.google_python,
    )


def _load_sheet_sync_preset_into_state(preset: SheetSyncPreset) -> None:
    st.session_state["sheet_sync_name"] = preset.name
    st.session_state["sheet_sync_spreadsheet_label"] = get_sheet_sync_spreadsheet_label(preset)
    st.session_state["sheet_sync_service_account_file"] = preset.service_account_file
    st.session_state["sheet_sync_spreadsheet_id"] = preset.spreadsheet_id
    st.session_state["sheet_sync_sheet_name"] = preset.sheet_name
    st.session_state["sheet_sync_url_column"] = preset.url_column
    st.session_state["sheet_sync_style_column"] = preset.style_column
    st.session_state["sheet_sync_transcription_column"] = preset.transcription_column
    st.session_state["sheet_sync_status_column"] = preset.status_column
    st.session_state["sheet_sync_processed_at_column"] = preset.processed_at_column
    st.session_state["sheet_sync_error_column"] = preset.error_column
    st.session_state["sheet_sync_audio_style"] = preset.audio_style
    st.session_state["sheet_sync_image_style"] = preset.image_style
    st.session_state["sheet_sync_google_python"] = preset.google_python


def _collect_sheet_sync_preset_from_state() -> SheetSyncPreset:
    return SheetSyncPreset(
        name=st.session_state.get("sheet_sync_name", "").strip(),
        spreadsheet_label=st.session_state.get("sheet_sync_spreadsheet_label", "").strip(),
        service_account_file=st.session_state.get("sheet_sync_service_account_file", "").strip(),
        spreadsheet_id=st.session_state.get("sheet_sync_spreadsheet_id", "").strip(),
        sheet_name=st.session_state.get("sheet_sync_sheet_name", "").strip(),
        url_column=st.session_state.get("sheet_sync_url_column", "G").strip() or "G",
        style_column=st.session_state.get("sheet_sync_style_column", "K").strip() or "K",
        transcription_column=st.session_state.get("sheet_sync_transcription_column", "L").strip() or "L",
        status_column=st.session_state.get("sheet_sync_status_column", "M").strip() or "M",
        processed_at_column=st.session_state.get("sheet_sync_processed_at_column", "N").strip() or "N",
        error_column=st.session_state.get("sheet_sync_error_column", "O").strip() or "O",
        audio_style=st.session_state.get("sheet_sync_audio_style", "音声リール").strip() or "音声リール",
        image_style=st.session_state.get("sheet_sync_image_style", "画像リール").strip() or "画像リール",
        google_python=st.session_state.get("sheet_sync_google_python", r"C:\Python313\python.exe").strip()
        or r"C:\Python313\python.exe",
    )


def _ensure_sheet_sync_dashboard_state() -> None:
    if "sheet_sync_presets" in st.session_state:
        return

    config_path = get_sheet_dashboard_config_path()
    presets, selected_name = load_sheet_sync_presets(config_path=config_path)
    preset_map = {preset.name: preset for preset in presets}
    selected_preset = preset_map[selected_name]
    st.session_state["sheet_sync_presets"] = preset_map
    st.session_state["sheet_sync_config_path"] = str(config_path)
    st.session_state["sheet_sync_selected_preset"] = selected_name
    st.session_state["sheet_sync_selected_spreadsheet"] = get_sheet_sync_spreadsheet_label(selected_preset)
    st.session_state["sheet_sync_loaded_preset"] = selected_name
    st.session_state["sheet_sync_loaded_spreadsheet"] = get_sheet_sync_spreadsheet_label(selected_preset)
    st.session_state["sheet_sync_last_output"] = ""
    st.session_state["sheet_sync_last_exit_code"] = None
    st.session_state["sheet_sync_last_command"] = ""
    st.session_state["sheet_sync_last_saved_name"] = ""
    st.session_state["sheet_sync_last_saved_spreadsheet"] = ""
    st.session_state["sheet_sync_last_saved_at"] = ""
    _load_sheet_sync_preset_into_state(selected_preset)


def _queue_sheet_sync_spreadsheet_selection(selected_name: str) -> None:
    st.session_state["sheet_sync_selected_spreadsheet"] = selected_name
    st.session_state[SHEET_SYNC_PENDING_SELECTED_SPREADSHEET_KEY] = selected_name


def _sync_sheet_sync_spreadsheet_selection(spreadsheet_options: list[str]) -> None:
    resolved = resolve_sheet_sync_selected_preset(
        spreadsheet_options,
        selected_name=st.session_state.get("sheet_sync_selected_spreadsheet", ""),
        widget_value=st.session_state.get(SHEET_SYNC_SELECTED_SPREADSHEET_WIDGET_KEY, ""),
        pending_name=st.session_state.pop(SHEET_SYNC_PENDING_SELECTED_SPREADSHEET_KEY, ""),
    )
    st.session_state["sheet_sync_selected_spreadsheet"] = resolved
    if resolved:
        st.session_state[SHEET_SYNC_SELECTED_SPREADSHEET_WIDGET_KEY] = resolved


def _queue_sheet_sync_preset_selection(selected_name: str) -> None:
    st.session_state["sheet_sync_selected_preset"] = selected_name
    st.session_state[SHEET_SYNC_PENDING_SELECTED_PRESET_KEY] = selected_name


def _sync_sheet_sync_preset_selection(preset_options: list[str]) -> None:
    resolved = resolve_sheet_sync_selected_preset(
        preset_options,
        selected_name=st.session_state.get("sheet_sync_selected_preset", ""),
        widget_value=st.session_state.get(SHEET_SYNC_SELECTED_PRESET_WIDGET_KEY, ""),
        pending_name=st.session_state.pop(SHEET_SYNC_PENDING_SELECTED_PRESET_KEY, ""),
    )
    st.session_state["sheet_sync_selected_preset"] = resolved
    if resolved:
        st.session_state[SHEET_SYNC_SELECTED_PRESET_WIDGET_KEY] = resolved


def _persist_sheet_sync_presets(preset_map: dict[str, SheetSyncPreset], selected_name: str) -> None:
    ordered_presets = list(preset_map.values())
    config_path = Path(st.session_state["sheet_sync_config_path"])
    save_sheet_sync_presets(
        presets=ordered_presets,
        selected_name=selected_name,
        config_path=config_path,
    )


def _sheet_sync_tab_cache_token(preset: SheetSyncPreset) -> str:
    return "|".join(
        [
            preset.google_python.strip(),
            preset.service_account_file.strip(),
            preset.spreadsheet_id.strip(),
        ]
    )


def _fetch_sheet_sync_tab_names(preset: SheetSyncPreset) -> list[str]:
    helper_args = [
        preset.google_python,
        "-m",
        "app.google_sheets_bridge",
        "list-sheets",
        "--service-account-file",
        preset.service_account_file,
        "--spreadsheet-id",
        preset.spreadsheet_id,
    ]
    completed = subprocess.run(
        helper_args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "対象タブ一覧の取得に失敗しました。"
        raise RuntimeError(message)

    payload = json.loads((completed.stdout or "").strip() or "{}")
    sheets = payload.get("sheets", []) if isinstance(payload, dict) else []
    tab_names = [
        str(sheet.get("title", "")).strip()
        for sheet in sheets
        if isinstance(sheet, dict) and str(sheet.get("title", "")).strip()
    ]
    seen: set[str] = set()
    ordered_names: list[str] = []
    for tab_name in tab_names:
        if tab_name in seen:
            continue
        seen.add(tab_name)
        ordered_names.append(tab_name)
    return ordered_names


def _get_sheet_sync_tab_names(preset: SheetSyncPreset, *, force_refresh: bool = False) -> tuple[list[str], str]:
    if (
        not preset.google_python.strip()
        or not preset.service_account_file.strip()
        or not preset.spreadsheet_id.strip()
    ):
        return [], ""

    google_python_path = Path(preset.google_python).expanduser()
    service_account_path = Path(preset.service_account_file).expanduser()
    if not google_python_path.exists() or not service_account_path.exists():
        return [], ""

    cache = st.session_state.setdefault(SHEET_SYNC_TAB_CACHE_KEY, {})
    cache_token = _sheet_sync_tab_cache_token(preset)
    if force_refresh or cache_token not in cache:
        try:
            cache[cache_token] = {
                "tab_names": _fetch_sheet_sync_tab_names(preset),
                "error": "",
            }
        except RuntimeError as exc:
            cache[cache_token] = {
                "tab_names": [],
                "error": str(exc),
            }

    entry = cache.get(cache_token, {})
    return list(entry.get("tab_names", [])), str(entry.get("error", ""))


def _render_sheet_sync_sheet_name_field(preset: SheetSyncPreset) -> None:
    tab_names, tab_error = _get_sheet_sync_tab_names(preset)

    if tab_names:
        choice_options = tab_names + [SHEET_SYNC_MANUAL_TAB_OPTION]
        resolved_choice = resolve_sheet_sync_selected_preset(
            choice_options,
            selected_name=(
                preset.sheet_name
                if preset.sheet_name in tab_names
                else SHEET_SYNC_MANUAL_TAB_OPTION
            ),
            widget_value=st.session_state.get(SHEET_SYNC_TAB_CHOICE_WIDGET_KEY, ""),
        )
        st.session_state[SHEET_SYNC_TAB_CHOICE_WIDGET_KEY] = resolved_choice

        select_col, refresh_col = st.columns([4, 1])
        selected_choice = select_col.selectbox(
            "対象タブ",
            options=choice_options,
            key=SHEET_SYNC_TAB_CHOICE_WIDGET_KEY,
        )
        if refresh_col.button("更新", use_container_width=True, key="sheet_sync_refresh_tabs"):
            _get_sheet_sync_tab_names(preset, force_refresh=True)
            st.rerun()
        if selected_choice != SHEET_SYNC_MANUAL_TAB_OPTION:
            st.session_state["sheet_sync_sheet_name"] = selected_choice
            st.caption(f"候補から選択中: {selected_choice}")
        else:
            st.text_input("対象タブ名", key="sheet_sync_sheet_name", placeholder="投稿データ260301")
        return

    st.text_input("対象タブ名", key="sheet_sync_sheet_name", placeholder="投稿データ260301")
    if tab_error:
        st.info("対象タブ一覧を取得できなかったため、手入力で指定してください。")
        with st.expander("対象タブ一覧の取得エラー"):
            st.code(tab_error)
        if st.button("対象タブ一覧を再取得", use_container_width=True, key="sheet_sync_retry_tabs"):
            _get_sheet_sync_tab_names(preset, force_refresh=True)
            st.rerun()


def _write_sheet_sync_log(
    log_path: Path,
    command: list[str],
    output: str,
    exit_code: int | str,
    *,
    started_at: str | None = None,
) -> None:
    ensure_directory(log_path.parent)
    payload = [
        "==== Shorts Visual Transcriber Sheet Sync Dashboard ====",
        f"Started: {started_at or time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Command: {subprocess.list2cmdline(command)}",
        f"Exit code: {exit_code}",
        "",
        output.strip(),
    ]
    log_path.write_text("\n".join(payload).strip() + "\n", encoding="utf-8")


def _append_sheet_sync_log_line(log_path: Path, line: str) -> None:
    ensure_directory(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def _run_sheet_sync_subprocess(command: list[str], google_python: str, live_placeholder) -> tuple[int, str]:
    env = os.environ.copy()
    env["SHEETS_SYNC_GOOGLE_PYTHON"] = google_python
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    idle_timeout_sec = float(env.get("SHEETS_SYNC_UI_IDLE_TIMEOUT_SEC", "240"))
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    _write_sheet_sync_log(
        SHEET_SYNC_LOG_PATH,
        command=command,
        output="",
        exit_code="running",
        started_at=started_at,
    )
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    assert process.stdout is not None
    last_output_at = time.monotonic()
    while True:
        line = process.stdout.readline()
        if line:
            last_output_at = time.monotonic()
            lines.append(line.rstrip("\n"))
            _append_sheet_sync_log_line(SHEET_SYNC_LOG_PATH, line)
            live_placeholder.code("\n".join(lines[-120:]) or "ログ待機中です...")
            continue
        if process.poll() is not None:
            break
        if time.monotonic() - last_output_at > idle_timeout_sec:
            timeout_line = f"[ERROR] Sheets subprocess idle timeout after {idle_timeout_sec:.0f}s without output."
            lines.append(timeout_line)
            _append_sheet_sync_log_line(SHEET_SYNC_LOG_PATH, timeout_line)
            live_placeholder.code("\n".join(lines[-120:]))
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(0.1)

    remainder = process.stdout.read()
    if remainder:
        lines.extend(remainder.splitlines())
        for line in remainder.splitlines():
            _append_sheet_sync_log_line(SHEET_SYNC_LOG_PATH, line)

    exit_code = process.wait()
    output = "\n".join(lines).strip()
    _write_sheet_sync_log(
        SHEET_SYNC_LOG_PATH,
        command=command,
        output=output,
        exit_code=exit_code,
        started_at=started_at,
    )
    return exit_code, output


def _render_sheet_sync_dashboard(
    *,
    output_dir: Path,
    download_dir: Path,
    cookies_file: Path | None,
    asr_model_size: str,
    asr_language: str,
    langs: str,
    ocr_backend: str,
    gpu: bool,
    keep_video: bool,
    sample_fps: float,
    scene_threshold: float,
    text_change_threshold: float,
    min_interval_sec: float,
    similarity_threshold: float,
    min_ocr_confidence: float,
) -> None:
    _ensure_sheet_sync_dashboard_state()

    preset_map: dict[str, SheetSyncPreset] = st.session_state["sheet_sync_presets"]
    grouped_presets = group_sheet_sync_presets(list(preset_map.values()))
    spreadsheet_options = list(grouped_presets.keys()) + ["新規スプレッドシート"]
    _sync_sheet_sync_spreadsheet_selection(spreadsheet_options)
    selected_spreadsheet = st.selectbox(
        "スプレッドシート",
        options=spreadsheet_options,
        key=SHEET_SYNC_SELECTED_SPREADSHEET_WIDGET_KEY,
    )
    st.session_state["sheet_sync_selected_spreadsheet"] = selected_spreadsheet

    current_group_presets = grouped_presets.get(selected_spreadsheet, [])
    preset_options = [preset.name for preset in current_group_presets] + ["新規設定"]
    _sync_sheet_sync_preset_selection(preset_options)
    selected_option = st.selectbox(
        "設定",
        options=preset_options,
        key=SHEET_SYNC_SELECTED_PRESET_WIDGET_KEY,
    )
    st.session_state["sheet_sync_selected_preset"] = selected_option

    if (
        st.session_state.get("sheet_sync_loaded_preset") != selected_option
        or st.session_state.get("sheet_sync_loaded_spreadsheet") != selected_spreadsheet
    ):
        if selected_spreadsheet == "新規スプレッドシート":
            preset = _new_sheet_sync_preset()
        elif selected_option == "新規設定":
            base_preset = current_group_presets[0] if current_group_presets else None
            preset = _new_sheet_sync_preset(
                base_preset=base_preset,
                spreadsheet_label=selected_spreadsheet,
            )
        else:
            preset = preset_map[selected_option]
        _load_sheet_sync_preset_into_state(preset)
        st.session_state["sheet_sync_loaded_preset"] = selected_option
        st.session_state["sheet_sync_loaded_spreadsheet"] = selected_spreadsheet
        st.rerun()

    st.caption(f"設定保存先: {st.session_state['sheet_sync_config_path']}")
    st.caption("通常は対象タブ名だけ設定すれば実行できます。名前変更や接続先の変更は「管理」、列やスタイルの変更は「詳細設定」を開いてください。")
    st.caption("実行時に、シートのヘッダーと先頭行から列配置とスタイル名を自動判定します。")
    st.caption("左サイドバーの OCR / ASR 設定と出力先が、そのままSheetsモードにも使われます。")

    summary_col1, summary_col2 = st.columns(2)
    summary_col1.caption(f"スプレッドシート: {selected_spreadsheet}")
    summary_col2.caption(f"設定: {selected_option}")

    with st.expander("管理"):
        st.caption("名前変更や接続先変更が必要なときだけ開いてください。")
        col1, col2 = st.columns(2)
        col1.text_input("スプレッドシート名", key="sheet_sync_spreadsheet_label", placeholder="例: インスタ運用シート")
        col2.text_input("設定名", key="sheet_sync_name", placeholder="例: 260301 本番")
        col1, col2 = st.columns(2)
        col1.text_input("Service Account JSON", key="sheet_sync_service_account_file")
        col2.text_input("スプレッドシートID", key="sheet_sync_spreadsheet_id")
        st.caption("既存設定は「スプレッドシート名」または「設定名」を変えて保存すると整理し直せます。")

    with st.expander("詳細設定"):
        col1, col2, col3 = st.columns(3)
        col1.text_input("URL列", key="sheet_sync_url_column")
        col2.text_input("音声スタイル名", key="sheet_sync_audio_style")
        col3.text_input("画像スタイル名", key="sheet_sync_image_style")
        col1, col2, col3, col4 = st.columns(4)
        col1.text_input("スタイル列", key="sheet_sync_style_column")
        col2.text_input("文字起こし列", key="sheet_sync_transcription_column")
        col3.text_input("状態列", key="sheet_sync_status_column")
        col4.text_input("処理日時列", key="sheet_sync_processed_at_column")
        col1, col2 = st.columns(2)
        col1.text_input("エラー列", key="sheet_sync_error_column")
        col2.text_input("Google API用 Python", key="sheet_sync_google_python")

    draft_preset = _collect_sheet_sync_preset_from_state()
    _render_sheet_sync_sheet_name_field(draft_preset)
    draft_preset = _collect_sheet_sync_preset_from_state()
    saved_preset = preset_map.get(selected_option) if selected_option in preset_map else None
    save_status_level, save_status_message = get_sheet_sync_save_status(
        draft_preset,
        selected_name=selected_option,
        saved_preset=saved_preset,
    )
    if save_status_level == "success":
        st.success(save_status_message)
    elif save_status_level == "warning":
        st.warning(save_status_message)
    else:
        st.info(save_status_message)
    if (
        save_status_level == "success"
        and st.session_state.get("sheet_sync_last_saved_name") == draft_preset.name
        and st.session_state.get("sheet_sync_last_saved_spreadsheet") == draft_preset.spreadsheet_label
        and st.session_state.get("sheet_sync_last_saved_at")
    ):
        st.caption(f"このセッションでは {st.session_state['sheet_sync_last_saved_at']} に保存しました。")
    if draft_preset.spreadsheet_id and not draft_preset.sheet_name:
        st.info("このスプレッドシートは追加済みです。対象タブ名を入力して保存すると実行できます。")

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("設定を保存", use_container_width=True):
        preset = _collect_sheet_sync_preset_from_state()
        if not preset.spreadsheet_label or not preset.name or not preset.service_account_file or not preset.spreadsheet_id:
            st.error("スプレッドシート名、設定名、Service Account JSON、スプレッドシートIDは必須です。")
        else:
            previous_name = selected_option if selected_option != "新規設定" else ""
            renamed = bool(
                previous_name and (
                    previous_name != preset.name
                    or selected_spreadsheet != preset.spreadsheet_label
                )
            )
            try:
                updated_map = upsert_sheet_sync_preset(
                    preset_map,
                    preset,
                    previous_name=previous_name,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["sheet_sync_presets"] = updated_map
                st.session_state["sheet_sync_loaded_preset"] = preset.name
                st.session_state["sheet_sync_loaded_spreadsheet"] = preset.spreadsheet_label
                st.session_state["sheet_sync_last_saved_name"] = preset.name
                st.session_state["sheet_sync_last_saved_spreadsheet"] = preset.spreadsheet_label
                st.session_state["sheet_sync_last_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _queue_sheet_sync_spreadsheet_selection(preset.spreadsheet_label)
                _queue_sheet_sync_preset_selection(preset.name)
                _persist_sheet_sync_presets(updated_map, selected_name=preset.name)
                st.success("設定名を変更しました。" if renamed else "設定を保存しました。")
                st.rerun()

    delete_disabled = (
        selected_spreadsheet == "新規スプレッドシート"
        or selected_option == "新規設定"
        or selected_option not in preset_map
    )
    if action_col2.button("選択中の設定を削除", disabled=delete_disabled, use_container_width=True):
        updated_map = dict(preset_map)
        updated_map.pop(selected_option, None)
        if not updated_map:
            fallback = default_sheet_sync_preset()
            updated_map = {fallback.name: fallback}
        next_grouped_presets = group_sheet_sync_presets(list(updated_map.values()))
        if selected_spreadsheet in next_grouped_presets:
            next_spreadsheet = selected_spreadsheet
            next_name = next_grouped_presets[selected_spreadsheet][0].name
        else:
            next_spreadsheet = next(iter(next_grouped_presets))
            next_name = next_grouped_presets[next_spreadsheet][0].name
        st.session_state["sheet_sync_presets"] = updated_map
        st.session_state["sheet_sync_loaded_preset"] = next_name
        st.session_state["sheet_sync_loaded_spreadsheet"] = next_spreadsheet
        _load_sheet_sync_preset_into_state(updated_map[next_name])
        _queue_sheet_sync_spreadsheet_selection(next_spreadsheet)
        _queue_sheet_sync_preset_selection(next_name)
        _persist_sheet_sync_presets(updated_map, selected_name=next_name)
        st.success("設定を削除しました。")
        st.rerun()

    st.divider()
    operation = st.radio(
        "実行内容",
        options=list(SHEET_SYNC_PRESET_OPTIONS.keys()),
        format_func=lambda value: SHEET_SYNC_PRESET_OPTIONS[value],
        horizontal=False,
    )

    current_preset = _collect_sheet_sync_preset_from_state()
    request = SheetSyncRunRequest(
        mode=operation,
        output_dir=str(output_dir),
        download_dir=str(download_dir),
        cookies_file=str(cookies_file) if cookies_file else "",
        audio_model_size=asr_model_size,
        audio_language=asr_language,
        langs=langs,
        ocr_backend=ocr_backend,
        min_ocr_confidence=min_ocr_confidence,
        gpu=gpu,
        sample_fps=sample_fps,
        scene_threshold=scene_threshold,
        text_change_threshold=text_change_threshold,
        min_interval_sec=min_interval_sec,
        similarity_threshold=similarity_threshold,
        keep_video=keep_video,
    )
    effective_ocr_backend = resolve_sheet_sync_ocr_backend(request)
    if operation == RUN_MODE_RERUN_LOW_QUALITY_OCR:
        st.info(
            "この実行では、完了済みの画像リールのうち空結果または低信頼なOCR出力だけを "
            "EasyOCR で順番に再実行します。"
        )
    elif ocr_backend == "easyocr" and effective_ocr_backend != ocr_backend:
        st.info(
            "Sheetsの一括実行では EasyOCR が重すぎるため、安定性優先で PaddleOCR に切り替えて実行します。"
            " EasyOCR を使うのは「未処理を1件だけテスト」のときだけにしてください。"
        )
    command = build_sheet_sync_command(
        python_executable=sys.executable,
        preset=current_preset,
        request=request,
    )

    with st.expander("デバッグ"):
        if st.button("最新ログを再読込", use_container_width=True, key="sheet_sync_reload_log_debug"):
            if SHEET_SYNC_LOG_PATH.exists():
                st.session_state["sheet_sync_last_output"] = read_text_file_best_effort(SHEET_SYNC_LOG_PATH)
                st.session_state["sheet_sync_last_exit_code"] = 0
            else:
                st.info("まだログがありません。")
        st.code(subprocess.list2cmdline(command), language="powershell")

    live_log_placeholder = st.empty()
    run_disabled = not current_preset.sheet_name or not current_preset.service_account_file or not current_preset.spreadsheet_id
    if run_disabled:
        st.caption("実行するには、対象タブ名・Service Account JSON・スプレッドシートID を設定して保存してください。")
    if st.button("Sheetsモードを実行", type="primary", use_container_width=True, disabled=run_disabled):
        service_account_path = Path(current_preset.service_account_file).expanduser()
        google_python_path = Path(current_preset.google_python).expanduser()
        if not current_preset.service_account_file or not service_account_path.exists():
            st.error("Service Account JSON のパスが正しくありません。")
        elif not current_preset.spreadsheet_id or not current_preset.sheet_name:
            st.error("スプレッドシートIDと対象タブ名を入力してください。")
        elif not current_preset.google_python or not google_python_path.exists():
            st.error("Google API用 Python のパスが正しくありません。")
        else:
            with st.spinner("Sheetsモードを実行しています"):
                exit_code, output = _run_sheet_sync_subprocess(
                    command=command,
                    google_python=current_preset.google_python,
                    live_placeholder=live_log_placeholder,
                )
            st.session_state["sheet_sync_last_output"] = output
            st.session_state["sheet_sync_last_exit_code"] = exit_code
            st.session_state["sheet_sync_last_command"] = subprocess.list2cmdline(command)

    last_output = st.session_state.get("sheet_sync_last_output", "")
    last_exit_code = st.session_state.get("sheet_sync_last_exit_code")
    if last_exit_code is not None:
        summary = summarize_sheet_sync_output(last_output)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("対象行", summary["rows"])
        col2.metric("完了", summary["done"])
        col3.metric("エラー", summary["errors"])
        col4.metric("終了コード", last_exit_code)
        if last_exit_code == 0:
            st.success("Sheetsモードの実行が完了しました。")
        else:
            st.error("Sheetsモードでエラーが発生しました。ログを確認してください。")

    if last_output:
        st.text_area("実行ログ", value=last_output, height=320, key="sheet_sync_last_output_view")
    elif SHEET_SYNC_LOG_PATH.exists():
        st.text_area(
            "最新ログ",
            value=read_text_file_best_effort(SHEET_SYNC_LOG_PATH),
            height=320,
            key="sheet_sync_log_view",
        )

    if SHEET_SYNC_LOG_PATH.exists():
        st.download_button(
            "ログをダウンロード",
            data=SHEET_SYNC_LOG_PATH.read_bytes(),
            file_name=SHEET_SYNC_LOG_PATH.name,
            mime="text/plain",
            key="sheet_sync_log_download",
        )



def main() -> None:
    st.set_page_config(page_title="ショート動画文字起こし", layout="wide")
    st.title("ショート動画 文字起こしツール")
    st.caption("Sheetsモード、Whisperモード、OCRモード を1つの画面から操作できます。")
    st.caption(f"実行用キャッシュ: {RUNTIME_ROOT}")

    with st.sidebar:
        st.header("設定")
        output_dir_str = st.text_input("出力フォルダ", value=str(DEFAULT_OUTPUT_DIR))
        download_dir_str = st.text_input("ダウンロードフォルダ", value=str(DEFAULT_DOWNLOAD_DIR))
        cookies_file_str = st.text_input("Cookiesファイルのパス（任意）", value="")
        asr_model_size = st.selectbox("Instagram音声認識モデル", options=["tiny", "base", "small"], index=2)
        asr_language = st.selectbox("Instagram音声言語", options=["auto", "ja", "en"], index=1)
        langs = st.text_input("OCR言語", value="ja,en")
        ocr_backend = st.selectbox(
            "OCR backend",
            options=["auto", "easyocr", "paddle"],
            index=0,
            format_func=lambda value: OCR_BACKEND_LABELS[value],
        )
        gpu = st.checkbox("OCRでGPUを使う", value=False)
        keep_video = st.checkbox("ダウンロード動画を残す", value=False)

        sample_fps = st.slider("解析FPS", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
        scene_threshold = st.slider("シーン変化しきい値", min_value=0.05, max_value=0.95, value=0.22, step=0.01)
        text_change_threshold = st.slider("文字変化しきい値", min_value=0.01, max_value=0.30, value=0.055, step=0.005)
        min_interval_sec = st.slider("最小間隔（秒）", min_value=0.1, max_value=5.0, value=0.4, step=0.1)
        similarity_threshold = st.slider("統合類似度しきい値", min_value=0.50, max_value=0.99, value=0.86, step=0.01)
        min_ocr_confidence = st.slider("OCR最低信頼度", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

    output_dir = Path(output_dir_str).expanduser().resolve()
    download_dir = Path(download_dir_str).expanduser().resolve()
    cookies_file = Path(cookies_file_str).expanduser().resolve() if cookies_file_str.strip() else None
    upload_dir = DEFAULT_UPLOAD_DIR
    ensure_directory(output_dir)
    ensure_directory(download_dir)
    ensure_directory(upload_dir)

    options = _build_processing_options(
        sample_fps=sample_fps,
        scene_threshold=scene_threshold,
        text_change_threshold=text_change_threshold,
        min_interval_sec=min_interval_sec,
        similarity_threshold=similarity_threshold,
        keep_video=keep_video,
        cookies_file=cookies_file,
    )

    sheet_tab, instagram_tab, general_tab = st.tabs(["Sheetsモード", "Whisperモード", "OCRモード"])

    with sheet_tab:
        st.subheader("Sheetsモード")
        st.write("設定を保存しておけば、シート名を切り替えてボタンだけで実行できます。")
        st.caption("未処理実行、1件テスト、エラー再実行、画像/音声リールの再実行に対応しています。")
        _render_sheet_sync_dashboard(
            output_dir=output_dir,
            download_dir=download_dir,
            cookies_file=cookies_file,
            asr_model_size=asr_model_size,
            asr_language=asr_language,
            langs=langs,
            ocr_backend=ocr_backend,
            gpu=gpu,
            keep_video=keep_video,
            sample_fps=sample_fps,
            scene_threshold=scene_threshold,
            text_change_threshold=text_change_threshold,
            min_interval_sec=min_interval_sec,
            similarity_threshold=similarity_threshold,
            min_ocr_confidence=min_ocr_confidence,
        )

    with instagram_tab:
        st.subheader("Whisperモード")
        st.write("URLを1本貼るだけで、音声認識した全文を上から確認できます。")
        st.caption("初回は音声認識モデルの準備に時間がかかることがあります。")

        with st.form("instagram_reel_form"):
            reel_url = st.text_input(
                "Instagram Reel URL",
                placeholder="https://www.instagram.com/reel/XXXXXXXXXXX/",
            )
            run_reel_clicked = st.form_submit_button("全文を文字起こしする", type="primary")

        instagram_status_placeholder = st.empty()
        instagram_details_placeholder = st.empty()
        instagram_progress_placeholder = st.empty()

        if run_reel_clicked:
            url = reel_url.strip()
            if not url:
                st.error("Instagram Reel URLを入力してください。")
            elif not _is_instagram_reel_url(url):
                st.error("Instagram ReelのURL形式ではありません。`https://www.instagram.com/reel/.../` を入力してください。")
            else:
                try:
                    with st.spinner("音声認識モデルを初期化しています"):
                        audio_transcriber = _build_audio_transcriber(
                            model_size=asr_model_size,
                            language_key=asr_language,
                        )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    audio_options = _build_audio_processing_options(
                        keep_video=keep_video,
                        cookies_file=cookies_file,
                    )
                    _run_audio_transcription(
                        session_prefix="instagram_reel",
                        inputs=[url],
                        audio_transcriber=audio_transcriber,
                        options=audio_options,
                        output_dir=output_dir,
                        download_dir=download_dir,
                        status_placeholder=instagram_status_placeholder,
                        details_placeholder=instagram_details_placeholder,
                        progress_placeholder=instagram_progress_placeholder,
                    )

        _render_saved_results(session_prefix="instagram_reel", show_full_text_first=True)

    with general_tab:
        st.subheader("OCRモード")

        with st.form("general_transcription_form"):
            raw_inputs = st.text_area(
                "URLまたはローカルパス（1行に1件）",
                placeholder=(
                    "https://www.instagram.com/reel/...\n"
                    "https://www.tiktok.com/@.../video/...\n"
                    "C:\\videos\\sample.mp4"
                ),
                height=160,
            )
            uploaded_files = st.file_uploader(
                "またはローカル動画ファイルをアップロード",
                type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
                accept_multiple_files=True,
            )
            run_general_clicked = st.form_submit_button("文字起こしを実行", type="primary")

        general_status_placeholder = st.empty()
        general_details_placeholder = st.empty()
        general_progress_placeholder = st.empty()

        if run_general_clicked:
            text_inputs = [line.strip() for line in raw_inputs.splitlines() if line.strip()]
            file_inputs = _save_uploaded_files(uploaded_files, upload_dir) if uploaded_files else []
            inputs = text_inputs + file_inputs

            if not inputs:
                st.error("URLまたはローカルパスを入力するか、動画ファイルをアップロードしてください。")
            else:
                try:
                    with st.spinner("OCRエンジンを初期化しています"):
                        ocr_engine = _build_ocr_engine(
                            languages_key=langs,
                            gpu=gpu,
                            min_confidence=min_ocr_confidence,
                            backend=ocr_backend,
                        )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    _run_transcription(
                        session_prefix="general",
                        inputs=inputs,
                        ocr_engine=ocr_engine,
                        options=options,
                        output_dir=output_dir,
                        download_dir=download_dir,
                        status_placeholder=general_status_placeholder,
                        details_placeholder=general_details_placeholder,
                        progress_placeholder=general_progress_placeholder,
                    )

        _render_saved_results(session_prefix="general", show_full_text_first=False)


if __name__ == "__main__":
    main()
