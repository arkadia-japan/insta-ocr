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
    RUN_MODE_RETRY_ERRORS,
    RUN_MODE_TEST_ONE,
    SheetSyncPreset,
    SheetSyncRunRequest,
    build_sheet_sync_command,
    default_sheet_sync_preset,
    get_sheet_dashboard_config_path,
    load_sheet_sync_presets,
    read_text_file_best_effort,
    save_sheet_sync_presets,
    summarize_sheet_sync_output,
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
SHEET_SYNC_PRESET_OPTIONS = {
    RUN_MODE_PENDING: "未処理をまとめて実行",
    RUN_MODE_TEST_ONE: "未処理を1件だけテスト",
    RUN_MODE_RETRY_ERRORS: "エラー行だけ再実行",
    RUN_MODE_RERUN_IMAGES: "画像リールを再実行",
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
def _build_ocr_engine(languages_key: str, gpu: bool, min_confidence: float) -> OcrEngine:
    languages = [token.strip() for token in languages_key.split(",") if token.strip()]
    return OcrEngine(languages=languages, gpu=gpu, min_confidence=min_confidence)


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


def _new_sheet_sync_preset() -> SheetSyncPreset:
    base = default_sheet_sync_preset()
    return SheetSyncPreset(
        name="",
        service_account_file=base.service_account_file,
        spreadsheet_id=base.spreadsheet_id,
        sheet_name=base.sheet_name,
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
    st.session_state["sheet_sync_presets"] = preset_map
    st.session_state["sheet_sync_config_path"] = str(config_path)
    st.session_state["sheet_sync_selected_preset"] = selected_name
    st.session_state["sheet_sync_loaded_preset"] = selected_name
    st.session_state["sheet_sync_last_output"] = ""
    st.session_state["sheet_sync_last_exit_code"] = None
    st.session_state["sheet_sync_last_command"] = ""
    _load_sheet_sync_preset_into_state(preset_map[selected_name])


def _persist_sheet_sync_presets(preset_map: dict[str, SheetSyncPreset], selected_name: str) -> None:
    ordered_presets = list(preset_map.values())
    config_path = Path(st.session_state["sheet_sync_config_path"])
    save_sheet_sync_presets(
        presets=ordered_presets,
        selected_name=selected_name,
        config_path=config_path,
    )


def _write_sheet_sync_log(log_path: Path, command: list[str], output: str, exit_code: int) -> None:
    ensure_directory(log_path.parent)
    payload = [
        "==== Shorts Visual Transcriber Sheet Sync Dashboard ====",
        f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Command: {subprocess.list2cmdline(command)}",
        f"Exit code: {exit_code}",
        "",
        output.strip(),
    ]
    log_path.write_text("\n".join(payload).strip() + "\n", encoding="utf-8")


def _run_sheet_sync_subprocess(command: list[str], google_python: str, live_placeholder) -> tuple[int, str]:
    env = os.environ.copy()
    env["SHEETS_SYNC_GOOGLE_PYTHON"] = google_python
    env["PYTHONUTF8"] = "1"

    lines: list[str] = []
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
    while True:
        line = process.stdout.readline()
        if line:
            lines.append(line.rstrip("\n"))
            live_placeholder.code("\n".join(lines[-120:]) or "ログ待機中です...")
            continue
        if process.poll() is not None:
            break
        time.sleep(0.1)

    remainder = process.stdout.read()
    if remainder:
        lines.extend(remainder.splitlines())

    exit_code = process.wait()
    output = "\n".join(lines).strip()
    _write_sheet_sync_log(SHEET_SYNC_LOG_PATH, command=command, output=output, exit_code=exit_code)
    return exit_code, output


def _render_sheet_sync_dashboard(
    *,
    output_dir: Path,
    download_dir: Path,
    cookies_file: Path | None,
    asr_model_size: str,
    asr_language: str,
    langs: str,
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
    preset_names = list(preset_map.keys())
    preset_options = preset_names + ["新規設定"]
    selected_option = st.selectbox(
        "設定プリセット",
        options=preset_options,
        key="sheet_sync_selected_preset",
    )

    if st.session_state.get("sheet_sync_loaded_preset") != selected_option:
        preset = _new_sheet_sync_preset() if selected_option == "新規設定" else preset_map[selected_option]
        _load_sheet_sync_preset_into_state(preset)
        st.session_state["sheet_sync_loaded_preset"] = selected_option
        st.rerun()

    st.caption(f"設定保存先: {st.session_state['sheet_sync_config_path']}")
    st.caption("左サイドバーの OCR / ASR 設定と出力先が、そのままSheetsモードにも使われます。")

    col1, col2 = st.columns(2)
    col1.text_input("設定名", key="sheet_sync_name", placeholder="例: 260301 本番")
    col2.text_input("対象タブ名", key="sheet_sync_sheet_name", placeholder="投稿データ260301")

    col1, col2 = st.columns(2)
    col1.text_input("Service Account JSON", key="sheet_sync_service_account_file")
    col2.text_input("スプレッドシートID", key="sheet_sync_spreadsheet_id")

    col1, col2, col3 = st.columns(3)
    col1.text_input("URL列", key="sheet_sync_url_column")
    col2.text_input("音声スタイル名", key="sheet_sync_audio_style")
    col3.text_input("画像スタイル名", key="sheet_sync_image_style")

    with st.expander("詳細設定"):
        col1, col2, col3, col4 = st.columns(4)
        col1.text_input("スタイル列", key="sheet_sync_style_column")
        col2.text_input("文字起こし列", key="sheet_sync_transcription_column")
        col3.text_input("状態列", key="sheet_sync_status_column")
        col4.text_input("処理日時列", key="sheet_sync_processed_at_column")
        col1, col2 = st.columns(2)
        col1.text_input("エラー列", key="sheet_sync_error_column")
        col2.text_input("Google API用 Python", key="sheet_sync_google_python")

    action_col1, action_col2, action_col3 = st.columns(3)
    if action_col1.button("設定を保存", use_container_width=True):
        preset = _collect_sheet_sync_preset_from_state()
        if not preset.name or not preset.service_account_file or not preset.spreadsheet_id or not preset.sheet_name:
            st.error("設定名、Service Account JSON、スプレッドシートID、対象タブ名は必須です。")
        else:
            previous_name = selected_option if selected_option != "新規設定" else ""
            updated_map = dict(preset_map)
            if previous_name and previous_name != preset.name:
                updated_map.pop(previous_name, None)
            updated_map[preset.name] = preset
            st.session_state["sheet_sync_presets"] = updated_map
            st.session_state["sheet_sync_selected_preset"] = preset.name
            st.session_state["sheet_sync_loaded_preset"] = preset.name
            _persist_sheet_sync_presets(updated_map, selected_name=preset.name)
            st.success("設定を保存しました。")
            st.rerun()

    delete_disabled = selected_option == "新規設定" or selected_option not in preset_map
    if action_col2.button("選択中の設定を削除", disabled=delete_disabled, use_container_width=True):
        updated_map = dict(preset_map)
        updated_map.pop(selected_option, None)
        if not updated_map:
            fallback = default_sheet_sync_preset()
            updated_map = {fallback.name: fallback}
        next_name = next(iter(updated_map))
        st.session_state["sheet_sync_presets"] = updated_map
        st.session_state["sheet_sync_selected_preset"] = next_name
        st.session_state["sheet_sync_loaded_preset"] = next_name
        _load_sheet_sync_preset_into_state(updated_map[next_name])
        _persist_sheet_sync_presets(updated_map, selected_name=next_name)
        st.success("設定を削除しました。")
        st.rerun()

    if action_col3.button("最新ログを再読込", use_container_width=True):
        if SHEET_SYNC_LOG_PATH.exists():
            st.session_state["sheet_sync_last_output"] = read_text_file_best_effort(SHEET_SYNC_LOG_PATH)
            st.session_state["sheet_sync_last_exit_code"] = 0
        else:
            st.info("まだログがありません。")

    st.divider()
    operation = st.radio(
        "実行内容",
        options=list(SHEET_SYNC_PRESET_OPTIONS.keys()),
        format_func=lambda value: SHEET_SYNC_PRESET_OPTIONS[value],
        horizontal=False,
    )

    current_preset = _collect_sheet_sync_preset_from_state()
    command = build_sheet_sync_command(
        python_executable=sys.executable,
        preset=current_preset,
        request=SheetSyncRunRequest(
            mode=operation,
            output_dir=str(output_dir),
            download_dir=str(download_dir),
            cookies_file=str(cookies_file) if cookies_file else "",
            audio_model_size=asr_model_size,
            audio_language=asr_language,
            langs=langs,
            min_ocr_confidence=min_ocr_confidence,
            gpu=gpu,
            sample_fps=sample_fps,
            scene_threshold=scene_threshold,
            text_change_threshold=text_change_threshold,
            min_interval_sec=min_interval_sec,
            similarity_threshold=similarity_threshold,
            keep_video=keep_video,
        ),
    )

    with st.expander("実行コマンドを見る"):
        st.code(subprocess.list2cmdline(command), language="powershell")

    live_log_placeholder = st.empty()
    if st.button("Sheetsモードを実行", type="primary", use_container_width=True):
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
