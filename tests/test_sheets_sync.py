import pytest

from app.sheets_sync import (
    DONE_EMPTY_STATUS,
    DONE_STATUS,
    ERROR_STATUS,
    PROCESSING_STATUS,
    SheetRow,
    SheetSyncConfig,
    build_fetch_range,
    column_index_to_letter,
    column_letter_to_index,
    should_process_row,
)


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
