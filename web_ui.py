from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.audio_pipeline import AudioProcessingOptions, run_single_audio_input
from app.audio_transcriber import AudioTranscriber
from app.downloader import DownloadError
from app.ocr_engine import OcrEngine
from app.pipeline import ProcessingOptions, run_single_input
from app.runtime_paths import get_runtime_download_dir, get_runtime_output_dir, get_runtime_upload_dir, get_runtime_root
from app.utils import ensure_directory, format_compact_time

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = get_runtime_root()
DEFAULT_OUTPUT_DIR = get_runtime_output_dir()
DEFAULT_DOWNLOAD_DIR = get_runtime_download_dir()
DEFAULT_UPLOAD_DIR = get_runtime_upload_dir()
INSTAGRAM_REEL_PATTERN = re.compile(r"^https?://(www\.)?instagram\.com/reel/[^/\s?#]+/?(?:\?.*)?$", re.IGNORECASE)


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



def main() -> None:
    st.set_page_config(page_title="ショート動画文字起こし", layout="wide")
    st.title("ショート動画 文字起こしツール")
    st.caption("Instagram Reel全文 は音声認識、汎用OCR は画面内テキスト抽出に対応します。")
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

    instagram_tab, general_tab = st.tabs(["Instagram Reel全文", "汎用OCR"])

    with instagram_tab:
        st.subheader("Instagram Reel URLから全文を作成")
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
        st.subheader("複数URL・ローカル動画にも対応")

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
