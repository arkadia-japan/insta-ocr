from __future__ import annotations

import argparse
from pathlib import Path

from .downloader import DownloadError
from .ocr_engine import OcrEngine
from .pipeline import ProcessingOptions, run_single_input
from .runtime_paths import get_runtime_download_dir, get_runtime_output_dir
from .utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    default_output_dir = str(get_runtime_output_dir())
    default_download_dir = str(get_runtime_download_dir())
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe text shown in short-form videos (Instagram Reels, TikTok, YouTube Shorts) "
            "by OCR on scene/key frames."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Video URLs and/or local video file paths.")
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help=f"Directory to store JSON/TXT/SRT files. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--download-dir",
        default=default_download_dir,
        help=f"Directory to store temporary downloaded videos. Default: {default_download_dir}",
    )
    parser.add_argument(
        "--cookies-file",
        default=None,
        help="Path to a Netscape cookies.txt file for platform login-required videos.",
    )
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Frames analyzed per second.")
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.22,
        help="Scene change sensitivity (higher = fewer scene captures).",
    )
    parser.add_argument(
        "--text-change-threshold",
        type=float,
        default=0.055,
        help="Text-region change sensitivity (lower = detect smaller text changes).",
    )
    parser.add_argument(
        "--min-interval-sec",
        type=float,
        default=0.4,
        help="Minimum interval between kept keyframes.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.86,
        help="Text similarity threshold for merging consecutive segments.",
    )
    parser.add_argument(
        "--langs",
        default="ja,en",
        help="OCR languages for easyocr (comma-separated). Default: ja,en",
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for OCR if available.")
    parser.add_argument(
        "--min-ocr-confidence",
        type=float,
        default=0.15,
        help="Minimum OCR confidence to keep detected text.",
    )
    parser.add_argument("--keep-video", action="store_true", help="Keep downloaded videos.")
    parser.add_argument("--json-indent", type=int, default=2, help="JSON indentation width.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    download_dir = Path(args.download_dir).resolve()
    ensure_directory(output_dir)
    ensure_directory(download_dir)

    langs = [token.strip() for token in args.langs.split(",") if token.strip()]
    if not langs:
        parser.error("At least one OCR language is required via --langs.")

    ocr_engine = OcrEngine(languages=langs, gpu=args.gpu, min_confidence=args.min_ocr_confidence)
    options = ProcessingOptions(
        sample_fps=args.sample_fps,
        scene_threshold=args.scene_threshold,
        text_change_threshold=args.text_change_threshold,
        min_interval_sec=args.min_interval_sec,
        similarity_threshold=args.similarity_threshold,
        json_indent=args.json_indent,
        keep_video=args.keep_video,
        cookies_file=Path(args.cookies_file).expanduser().resolve() if args.cookies_file else None,
        runtime_video_dir=download_dir,
    )

    failures = 0
    for input_ref in args.inputs:
        print(f"[INFO] Processing: {input_ref}")
        try:
            result, output_files = run_single_input(
                input_ref=input_ref,
                ocr_engine=ocr_engine,
                options=options,
                output_dir=output_dir,
                download_dir=download_dir,
            )
            print(
                "[DONE] "
                f"segments={len(result.segments)} sampled_segments={result.sampled_frames} "
                f"ocr_hits={result.ocr_hits} fallback_used={result.fallback_used} "
                f"json={output_files['json']}"
            )
            if not result.segments:
                print(
                    "[WARN] No text detected. Try lowering --min-ocr-confidence, "
                    "raising --sample-fps, or passing --langs that match the video text."
                )
        except (FileNotFoundError, DownloadError, RuntimeError) as exc:
            failures += 1
            print(f"[ERROR] {input_ref}: {exc}")

    return 1 if failures else 0
