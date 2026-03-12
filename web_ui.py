from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.downloader import DownloadError
from app.ocr_engine import OcrEngine
from app.pipeline import ProcessingOptions, run_single_input
from app.utils import ensure_directory, format_compact_time

PROJECT_ROOT = Path(__file__).resolve().parent


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


@st.cache_resource(show_spinner=False)
def _build_ocr_engine(languages_key: str, gpu: bool, min_confidence: float) -> OcrEngine:
    languages = [token.strip() for token in languages_key.split(",") if token.strip()]
    return OcrEngine(languages=languages, gpu=gpu, min_confidence=min_confidence)


def _make_status_reporter(status_box, details_box, input_ref: str, index: int, total: int):
    def report(message: str) -> None:
        status_box.update(label=f"{index}/{total} 件目を処理中", state="running")
        details_box.info(f"{input_ref}\n{message}")

    return report


def main() -> None:
    st.set_page_config(page_title="ショート動画 文字起こし", layout="wide")
    st.title("ショート動画 文字起こしツール（Web UI）")
    st.caption("音声なし・BGMのみのショート動画向けOCR文字起こし")

    with st.sidebar:
        st.header("設定")
        output_dir_str = st.text_input("出力フォルダ", value=str(PROJECT_ROOT / "output"))
        download_dir_str = st.text_input("ダウンロードフォルダ", value=str(PROJECT_ROOT / "downloads"))
        cookies_file_str = st.text_input("Cookiesファイルのパス（任意）", value="")
        langs = st.text_input("OCR言語", value="ja,en")
        gpu = st.checkbox("OCRでGPUを使用", value=False)
        keep_video = st.checkbox("ダウンロード動画を保持", value=False)

        sample_fps = st.slider("解析FPS", min_value=0.5, max_value=8.0, value=4.0, step=0.1)
        scene_threshold = st.slider("シーン変化しきい値", min_value=0.05, max_value=0.95, value=0.22, step=0.01)
        text_change_threshold = st.slider("文字変化しきい値", min_value=0.01, max_value=0.30, value=0.055, step=0.005)
        min_interval_sec = st.slider("最小間隔（秒）", min_value=0.1, max_value=5.0, value=0.4, step=0.1)
        similarity_threshold = st.slider(
            "類似度しきい値",
            min_value=0.50,
            max_value=0.99,
            value=0.86,
            step=0.01,
        )
        min_ocr_confidence = st.slider(
            "OCR最小信頼度",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
        )

    with st.form("transcription_form"):
        raw_inputs = st.text_area(
            "URL またはローカルパス（1行に1件）",
            placeholder=(
                "https://www.instagram.com/reel/...\n"
                "https://www.tiktok.com/@.../video/...\n"
                "C:\\videos\\sample.mp4"
            ),
            height=140,
        )
        uploaded_files = st.file_uploader(
            "またはローカル動画ファイルをアップロード",
            type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
            accept_multiple_files=True,
        )
        run_clicked = st.form_submit_button("文字起こしを実行", type="primary")

    status_placeholder = st.empty()
    details_placeholder = st.empty()
    progress_placeholder = st.empty()
    if run_clicked:
        status_box = status_placeholder.status("入力を確認しています", expanded=True)
        details_box = details_placeholder.empty()
        output_dir = Path(output_dir_str).expanduser().resolve()
        download_dir = Path(download_dir_str).expanduser().resolve()
        cookies_file = Path(cookies_file_str).expanduser().resolve() if cookies_file_str.strip() else None
        upload_dir = download_dir / "uploaded"
        ensure_directory(output_dir)
        ensure_directory(download_dir)

        text_inputs = [line.strip() for line in raw_inputs.splitlines() if line.strip()]
        file_inputs = _save_uploaded_files(uploaded_files, upload_dir) if uploaded_files else []
        inputs = text_inputs + file_inputs

        if not inputs:
            status_box.update(label="入力がありません", state="error")
            st.error("URL/パスを1件以上入力するか、動画ファイルをアップロードしてください。")
            st.stop()

        try:
            status_box.update(label="OCRエンジンを初期化しています", state="running")
            with st.spinner("OCRエンジンを初期化しています"):
                ocr_engine = _build_ocr_engine(
                    languages_key=langs,
                    gpu=gpu,
                    min_confidence=min_ocr_confidence,
                )
        except RuntimeError as exc:
            status_box.update(label="OCRエンジン初期化に失敗しました", state="error")
            st.error(str(exc))
            st.stop()

        options = ProcessingOptions(
            sample_fps=sample_fps,
            scene_threshold=scene_threshold,
            text_change_threshold=text_change_threshold,
            min_interval_sec=min_interval_sec,
            similarity_threshold=similarity_threshold,
            keep_video=keep_video,
            cookies_file=cookies_file,
        )

        success_records: list[dict] = []
        failures: list[str] = []
        progress = progress_placeholder.progress(0.0)

        for idx, input_ref in enumerate(inputs, start=1):
            reporter = _make_status_reporter(
                status_box=status_box,
                details_box=details_box,
                input_ref=input_ref,
                index=idx,
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
            progress.progress(idx / len(inputs))

        st.session_state["success_records"] = success_records
        st.session_state["failures"] = failures
        st.session_state["output_dir"] = str(output_dir)
        if failures:
            status_box.update(label="一部の処理で失敗しました", state="error")
        else:
            status_box.update(label="文字起こしが完了しました", state="complete")

    success_records = st.session_state.get("success_records", [])
    failures = st.session_state.get("failures", [])
    output_dir_state = st.session_state.get("output_dir")

    if output_dir_state:
        st.info(f"出力先: {output_dir_state}")

    if failures:
        st.error("一部の入力で処理に失敗しました。")
        for message in failures:
            st.write(f"- {message}")

    if not success_records:
        return

    st.success(f"{len(success_records)} 件の処理が完了しました。")
    for index, record in enumerate(success_records, start=1):
        result = record["result"]
        paths = record["paths"]

        st.subheader(f"{index}. {result.input_ref}")
        col1, col2, col3 = st.columns(3)
        col1.metric("セグメント数", len(result.segments))
        col2.metric("解析セグメント数", result.sampled_frames)
        col3.metric("動画長", format_compact_time(result.duration_sec))
        st.caption(f"OCRヒット: {result.ocr_hits} / fallback: {result.fallback_used}")

        if not result.segments:
            st.warning(
                "テキストが検出されませんでした。抽出設定（サンプリングFPS・文字変化しきい値）や"
                "OCR設定（信頼度・言語）を下げて再実行してください。"
            )

        segment_rows = [
            {
                "開始": format_compact_time(segment.start_sec),
                "終了": format_compact_time(segment.end_sec),
                "テキスト": segment.text,
                "信頼度": segment.confidence,
            }
            for segment in result.segments
        ]
        st.dataframe(segment_rows, use_container_width=True)

        json_path = Path(paths["json"])
        txt_path = Path(paths["txt"])
        srt_path = Path(paths["srt"])

        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "JSONをダウンロード",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
            key=f"json_{index}",
        )
        c2.download_button(
            "TXTをダウンロード",
            data=txt_path.read_bytes(),
            file_name=txt_path.name,
            mime="text/plain",
            key=f"txt_{index}",
        )
        c3.download_button(
            "SRTをダウンロード",
            data=srt_path.read_bytes(),
            file_name=srt_path.name,
            mime="application/x-subrip",
            key=f"srt_{index}",
        )

        with st.expander("JSONを表示"):
            st.code(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
