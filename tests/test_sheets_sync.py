import json
import subprocess
from pathlib import Path

import pytest

from app.models import TranscriptionResult, TranscriptSegment
from app.sheets_sync import (
    DONE_EMPTY_STATUS,
    DONE_STATUS,
    ERROR_STATUS,
    PROCESSING_STATUS,
    SheetRow,
    SheetsBridgeContext,
    SheetSyncConfig,
    assess_low_quality_sheet_row,
    _run_google_sheets_helper,
    _build_audio_error_retry_row,
    _build_done_empty_image_retry_row,
    _build_empty_audio_retry_row,
    build_fetch_range,
    column_index_to_letter,
    column_letter_to_index,
    infer_sheet_sync_config_from_preview,
    iter_pending_rows,
    should_process_row,
)
from app.utils import safe_stem_from_input


def _write_sheet_output_json(tmp_path, input_ref: str, segments: list[dict]) -> None:
    output_path = tmp_path / f"{safe_stem_from_input(input_ref)}.json"
    output_path.write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")


def test_column_conversion_round_trip() -> None:
    assert column_letter_to_index("A") == 0
    assert column_letter_to_index("K") == 10
    assert column_letter_to_index("AA") == 26
    assert column_index_to_letter(0) == "A"
    assert column_index_to_letter(10) == "K"
    assert column_index_to_letter(26) == "AA"


def test_build_fetch_range_uses_largest_configured_column() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="投稿データ260301",
        url_column="B",
        style_column="K",
        transcription_column="L",
        status_column="M",
        processed_at_column="N",
        error_column="O",
        start_row=2,
    )

    assert build_fetch_range(config) == "'投稿データ260301'!A2:O"


def test_infer_sheet_sync_config_from_preview_detects_headers_and_styles() -> None:
    config = SheetSyncConfig(spreadsheet_id="sheet-id", sheet_name="恋愛_260304", url_column="G")

    detected, messages = infer_sheet_sync_config_from_preview(
        config=config,
        preview_rows=[
            ["TikTok運用シート"],
            ["投稿日", "担当", "動画URL", "投稿タイプ", "文字起こし", "処理ステータス", "処理日時", "エラー内容"],
            ["2026/03/04", "A", "https://www.tiktok.com/@sample/video/1", "音声", "", "未処理", "", ""],
            ["2026/03/05", "B", "https://www.tiktok.com/@sample/video/2", "画像", "", "", "", ""],
        ],
    )

    assert detected == SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="C",
        style_column="D",
        transcription_column="E",
        status_column="F",
        processed_at_column="G",
        error_column="H",
        start_row=3,
        audio_style="音声",
        image_style="画像",
    )
    assert any("columns" in message for message in messages)
    assert any("styles" in message for message in messages)


def test_infer_sheet_sync_config_from_preview_can_use_sample_rows_when_headers_differ() -> None:
    config = SheetSyncConfig(spreadsheet_id="sheet-id", sheet_name="Shorts_260304", url_column="G")

    detected, _ = infer_sheet_sync_config_from_preview(
        config=config,
        preview_rows=[
            ["案件", "媒体", "リンク先", "処理モード", "メモ", "進行状況"],
            ["a", "YouTube", "https://www.youtube.com/shorts/123", "画像", "", "未処理"],
            ["b", "YouTube", "https://www.youtube.com/shorts/456", "音声", "", "再実行"],
        ],
    )

    assert detected.url_column == "C"
    assert detected.style_column == "D"
    assert detected.status_column == "F"


def test_infer_sheet_sync_config_from_preview_prefers_youtube_video_urls_over_thumbnails() -> None:
    config = SheetSyncConfig(spreadsheet_id="sheet-id", sheet_name="好き避け 女性_260304", url_column="G")

    detected, _ = infer_sheet_sync_config_from_preview(
        config=config,
        preview_rows=[
            ["状態", "メモ", "サムネURL", "動画URL", "投稿タイプ"],
            ["", "", "https://i.ytimg.com/vi/abc123/default.jpg", "https://www.youtube.com/shorts/abc123", "音声リール"],
            ["未処理", "", "https://i.ytimg.com/vi/def456/default.jpg", "https://www.youtube.com/shorts/def456", "音声リール"],
        ],
    )

    assert detected.url_column == "D"
    assert detected.style_column == "E"
    assert detected.status_column == "A"


def test_infer_sheet_sync_config_from_preview_detects_sparse_tiktok_headers() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="G",
        style_column="K",
        status_column="M",
    )

    def build_row(cells: dict[int, str]) -> list[str]:
        row = [""] * 23
        for index, value in cells.items():
            row[index] = value
        return row

    detected, messages = infer_sheet_sync_config_from_preview(
        config=config,
        preview_rows=[
            build_row(
                {
                    0: "サムネ",
                    1: "概要欄",
                    2: "再生数",
                    3: "いいね数",
                    4: "シェア数",
                    5: "コメント数",
                    6: "保存数",
                    7: "投稿日",
                    8: "URL",
                    9: "拡散率",
                    10: "プロフ写真",
                    11: "アカウント名",
                    12: "ユーザーID",
                    13: "フォロワー数",
                    14: "author_heart_count",
                    15: "投稿数",
                    16: "user_count",
                }
            ),
            build_row(
                {
                    0: "https://cdn.example.com/thumb1.jpg",
                    8: "https://www.tiktok.com/@sample/video/1",
                    10: "https://cdn.example.com/avatar1.jpg",
                    12: "ren_i66",
                }
            ),
            build_row(
                {
                    0: "https://cdn.example.com/thumb2.jpg",
                    8: "https://www.tiktok.com/@sample/video/2",
                    10: "https://cdn.example.com/avatar2.jpg",
                    12: "miraienomiti",
                }
            ),
        ],
    )

    assert detected.start_row == 2
    assert detected.url_column == "I"
    assert detected.style_column == "R"
    assert detected.transcription_column == "S"
    assert detected.status_column == "T"
    assert detected.processed_at_column == "U"
    assert detected.error_column == "V"
    assert detected.audio_style == "音声リール"
    assert detected.image_style == "画像リール"
    assert detected.default_style == "音声リール"
    assert any("columns" in message for message in messages)
    assert any("allocated columns" in message for message in messages)


def test_iter_pending_rows_uses_default_style_for_tiktok_and_shorts_rows() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="I",
        style_column="R",
        transcription_column="S",
        status_column="T",
        processed_at_column="U",
        error_column="V",
        audio_style="音声リール",
        image_style="画像リール",
        default_style="音声リール",
    )

    values = [
        [""] * 8 + ["https://www.tiktok.com/@sample/video/1"],
        [""] * 8 + ["https://www.youtube.com/shorts/abc"],
    ]

    rows = iter_pending_rows(values=values, config=config)

    assert [row.style for row in rows] == ["音声リール", "音声リール"]


def test_build_empty_audio_retry_row_switches_audio_empty_result_to_image() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="I",
        audio_style="音声リール",
        image_style="画像リール",
    )
    row = SheetRow(
        row_number=2,
        input_ref="https://www.tiktok.com/@sample/video/1",
        style="音声リール",
        transcription="",
        status=PROCESSING_STATUS,
        error="",
    )
    empty_result = TranscriptionResult(
        input_ref=row.input_ref,
        platform="tiktok",
        video_path="video.mp4",
        duration_sec=10.0,
        sampled_frames=0,
        ocr_hits=0,
        fallback_used=False,
        segments=[],
        transcription_mode="audio",
    )

    retry_row = _build_empty_audio_retry_row(row=row, config=config, result=empty_result)

    assert retry_row == SheetRow(
        row_number=2,
        input_ref="https://www.tiktok.com/@sample/video/1",
        style="画像リール",
        transcription="",
        status=PROCESSING_STATUS,
        error="",
    )


def test_build_empty_audio_retry_row_skips_non_empty_audio_result() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="I",
        audio_style="音声リール",
        image_style="画像リール",
    )
    row = SheetRow(
        row_number=2,
        input_ref="https://www.tiktok.com/@sample/video/1",
        style="音声リール",
        transcription="",
        status=PROCESSING_STATUS,
        error="",
    )
    audio_result = TranscriptionResult(
        input_ref=row.input_ref,
        platform="tiktok",
        video_path="video.mp4",
        duration_sec=10.0,
        sampled_frames=0,
        ocr_hits=0,
        fallback_used=False,
        segments=[TranscriptSegment(start_sec=0.0, end_sec=1.0, text="hello")],
        transcription_mode="audio",
    )

    retry_row = _build_empty_audio_retry_row(row=row, config=config, result=audio_result)

    assert retry_row is None


def test_build_audio_error_retry_row_switches_no_audio_stream_to_image() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="I",
        audio_style="音声リール",
        image_style="画像リール",
    )
    row = SheetRow(
        row_number=2,
        input_ref="https://www.youtube.com/shorts/abc",
        style="音声リール",
        transcription="",
        status=PROCESSING_STATUS,
        error="",
    )

    retry_row = _build_audio_error_retry_row(
        row=row,
        config=config,
        exc=RuntimeError("Audio transcription failed: no audio stream was found in the input video."),
    )

    assert retry_row == SheetRow(
        row_number=2,
        input_ref="https://www.youtube.com/shorts/abc",
        style="画像リール",
        transcription="",
        status=PROCESSING_STATUS,
        error="",
    )


def test_build_done_empty_image_retry_row_switches_existing_empty_audio_row() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="I",
        audio_style="音声リール",
        image_style="画像リール",
    )
    row = SheetRow(
        row_number=5,
        input_ref="https://www.tiktok.com/@sample/video/5",
        style="音声リール",
        transcription="",
        status=DONE_EMPTY_STATUS,
        error="",
    )

    retry_row = _build_done_empty_image_retry_row(row=row, config=config)

    assert retry_row == SheetRow(
        row_number=5,
        input_ref="https://www.tiktok.com/@sample/video/5",
        style="画像リール",
        transcription="",
        status=DONE_EMPTY_STATUS,
        error="",
    )


def test_iter_pending_rows_retries_existing_done_empty_audio_rows_as_image() -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="恋愛_260304",
        url_column="I",
        style_column="R",
        transcription_column="S",
        status_column="T",
        processed_at_column="U",
        error_column="V",
        audio_style="音声リール",
        image_style="画像リール",
    )

    values = [
        [""] * 8 + ["https://www.tiktok.com/@sample/video/1"] + [""] * 8 + ["音声リール", "", DONE_EMPTY_STATUS],
    ]

    rows = iter_pending_rows(values=values, config=config)

    assert rows == [
        SheetRow(
            row_number=2,
            input_ref="https://www.tiktok.com/@sample/video/1",
            style="画像リール",
            transcription="",
            status=DONE_EMPTY_STATUS,
            error="",
        )
    ]


def test_assess_low_quality_sheet_row_flags_low_confidence_output(tmp_path) -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="sheet",
        url_column="I",
        audio_style="audio",
        image_style="image",
    )
    row = SheetRow(
        row_number=2,
        input_ref="https://www.tiktok.com/@sample/video/2",
        style="image",
        transcription="some existing text",
        status=DONE_STATUS,
        error="",
    )
    _write_sheet_output_json(
        tmp_path,
        row.input_ref,
        [
            {"text": "bad text", "confidence": 0.51},
            {"text": "still bad", "confidence": 0.64},
        ],
    )

    should_rerun, reason = assess_low_quality_sheet_row(row=row, config=config, output_dir=tmp_path)

    assert should_rerun is True
    assert "avg_conf" in reason


def test_assess_low_quality_sheet_row_accepts_high_confidence_output(tmp_path) -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="sheet",
        url_column="I",
        audio_style="audio",
        image_style="image",
    )
    row = SheetRow(
        row_number=2,
        input_ref="https://www.tiktok.com/@sample/video/3",
        style="image",
        transcription="十分に長い文字起こしテキストです",
        status=DONE_STATUS,
        error="",
    )
    _write_sheet_output_json(
        tmp_path,
        row.input_ref,
        [
            {"text": "十分に長い文字起こしテキストです", "confidence": 0.96},
            {"text": "補足の行です", "confidence": 0.93},
        ],
    )

    should_rerun, reason = assess_low_quality_sheet_row(row=row, config=config, output_dir=tmp_path)

    assert should_rerun is False
    assert reason == "quality looks acceptable"


def test_iter_pending_rows_selects_only_low_quality_completed_image_rows(tmp_path) -> None:
    config = SheetSyncConfig(
        spreadsheet_id="sheet-id",
        sheet_name="sheet",
        url_column="I",
        style_column="R",
        transcription_column="S",
        status_column="T",
        processed_at_column="U",
        error_column="V",
        audio_style="audio",
        image_style="image",
    )
    low_quality_url = "https://www.tiktok.com/@sample/video/10"
    high_quality_url = "https://www.tiktok.com/@sample/video/11"
    _write_sheet_output_json(
        tmp_path,
        low_quality_url,
        [
            {"text": "bad text", "confidence": 0.58},
            {"text": "still bad", "confidence": 0.62},
        ],
    )
    _write_sheet_output_json(
        tmp_path,
        high_quality_url,
        [
            {"text": "十分に長い文字起こしテキストです", "confidence": 0.95},
            {"text": "補足の行です", "confidence": 0.91},
        ],
    )
    values = [
        [""] * 8 + [low_quality_url] + [""] * 8 + ["image", "existing text", DONE_STATUS],
        [""] * 8 + [high_quality_url] + [""] * 8 + ["image", "十分に長い文字起こしテキストです", DONE_STATUS],
    ]

    rows = iter_pending_rows(
        values=values,
        config=config,
        only_style="image",
        rerun_low_quality_ocr=True,
        quality_output_dir=tmp_path,
    )

    assert rows == [
        SheetRow(
            row_number=2,
            input_ref=low_quality_url,
            style="image",
            transcription="existing text",
            status=DONE_STATUS,
            error="",
        )
    ]


def test_run_google_sheets_helper_raises_timeout_error(monkeypatch) -> None:
    sheets = SheetsBridgeContext(
        service_account_file=Path(r"C:\secure\svc.json"),
        spreadsheet_id="sheet-id",
        google_python=r"C:\Python313\python.exe",
    )

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=45)

    monkeypatch.setattr("app.sheets_sync.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        _run_google_sheets_helper(sheets, "fetch-values")


@pytest.mark.parametrize(
    ("status", "retry_errors", "reprocess_existing", "expected"),
    [
        ("", False, False, True),
        ("未処理", False, False, True),
        ("再実行", False, False, True),
        (PROCESSING_STATUS, False, False, False),
        (DONE_STATUS, False, False, False),
        (DONE_EMPTY_STATUS, False, False, False),
        (ERROR_STATUS, False, False, False),
        (ERROR_STATUS, True, False, True),
        (DONE_STATUS, False, True, True),
        (DONE_EMPTY_STATUS, False, True, True),
        (ERROR_STATUS, False, True, True),
    ],
)
def test_should_process_row_status_rules(
    status: str,
    retry_errors: bool,
    reprocess_existing: bool,
    expected: bool,
) -> None:
    config = SheetSyncConfig(spreadsheet_id="sheet-id", sheet_name="投稿データ260301", url_column="J")
    row = SheetRow(
        row_number=2,
        input_ref="https://www.instagram.com/reel/example/",
        style="音声リール",
        transcription="",
        status=status,
        error="",
    )

    assert (
        should_process_row(
            row=row,
            config=config,
            retry_errors=retry_errors,
            reprocess_existing=reprocess_existing,
        )
        is expected
    )


def test_should_process_row_rejects_completed_or_unsupported_rows() -> None:
    config = SheetSyncConfig(spreadsheet_id="sheet-id", sheet_name="投稿データ260301", url_column="J")

    unsupported = SheetRow(
        row_number=2,
        input_ref="https://www.instagram.com/reel/example/",
        style="動画",
        transcription="",
        status="",
        error="",
    )
    completed = SheetRow(
        row_number=3,
        input_ref="https://www.instagram.com/reel/example/",
        style="画像リール",
        transcription="already done",
        status="",
        error="",
    )

    assert should_process_row(row=unsupported, config=config) is False
    assert should_process_row(row=completed, config=config) is False


def test_should_process_row_respects_style_filter() -> None:
    config = SheetSyncConfig(spreadsheet_id="sheet-id", sheet_name="投稿データ260301", url_column="J")
    row = SheetRow(
        row_number=4,
        input_ref="https://www.instagram.com/reel/example/",
        style="画像リール",
        transcription="already done",
        status=DONE_STATUS,
        error="",
    )

    assert should_process_row(row=row, config=config, reprocess_existing=True, only_style="画像リール") is True
    assert should_process_row(row=row, config=config, reprocess_existing=True, only_style="音声リール") is False
