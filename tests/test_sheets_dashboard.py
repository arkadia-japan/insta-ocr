from pathlib import Path

from app.sheets_dashboard import (
    RUN_MODE_RERUN_IMAGES,
    RUN_MODE_RETRY_ERRORS,
    RUN_MODE_TEST_ONE,
    SheetSyncPreset,
    SheetSyncRunRequest,
    build_sheet_sync_command,
    load_sheet_sync_presets,
    read_text_file_best_effort,
    save_sheet_sync_presets,
    summarize_sheet_sync_output,
)


def test_load_sheet_sync_presets_returns_default_when_missing(tmp_path: Path) -> None:
    presets, selected_name = load_sheet_sync_presets(tmp_path / "missing.json")

    assert len(presets) == 1
    assert selected_name == presets[0].name


def test_save_and_load_sheet_sync_presets_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "sheet_sync_presets.json"
    presets = [
        SheetSyncPreset(
            name="A",
            service_account_file=r"C:\secure\a.json",
            spreadsheet_id="sheet-a",
            sheet_name="投稿データA",
            url_column="G",
        ),
        SheetSyncPreset(
            name="B",
            service_account_file=r"C:\secure\b.json",
            spreadsheet_id="sheet-b",
            sheet_name="投稿データB",
            url_column="H",
        ),
    ]

    save_sheet_sync_presets(presets=presets, selected_name="B", config_path=config_path)
    loaded_presets, selected_name = load_sheet_sync_presets(config_path)

    assert loaded_presets == presets
    assert selected_name == "B"


def test_build_sheet_sync_command_adds_mode_specific_flags() -> None:
    preset = SheetSyncPreset(
        name="運用",
        service_account_file=r"C:\secure\svc.json",
        spreadsheet_id="spreadsheet-id",
        sheet_name="投稿データ260301",
        url_column="G",
    )
    request = SheetSyncRunRequest(
        mode=RUN_MODE_RERUN_IMAGES,
        output_dir=r"C:\runtime\output",
        download_dir=r"C:\runtime\downloads",
        cookies_file=r"C:\runtime\cookies.txt",
        gpu=True,
        keep_video=True,
    )

    command = build_sheet_sync_command("python", preset, request)

    assert command[:3] == ["python", "-m", "app.sheets_sync"]
    assert "--cookies-file" in command
    assert "--gpu" in command
    assert "--keep-video" in command
    assert command[-3:] == ["--only-style", "画像リール", "--reprocess-existing"]


def test_build_sheet_sync_command_supports_limit_and_retry_modes() -> None:
    preset = SheetSyncPreset(
        name="運用",
        service_account_file=r"C:\secure\svc.json",
        spreadsheet_id="spreadsheet-id",
        sheet_name="投稿データ260301",
        url_column="G",
    )

    test_one = build_sheet_sync_command(
        "python",
        preset,
        SheetSyncRunRequest(
            mode=RUN_MODE_TEST_ONE,
            output_dir=r"C:\runtime\output",
            download_dir=r"C:\runtime\downloads",
        ),
    )
    retry = build_sheet_sync_command(
        "python",
        preset,
        SheetSyncRunRequest(
            mode=RUN_MODE_RETRY_ERRORS,
            output_dir=r"C:\runtime\output",
            download_dir=r"C:\runtime\downloads",
        ),
    )

    assert test_one[-2:] == ["--limit", "1"]
    assert retry[-1] == "--retry-errors"


def test_summarize_sheet_sync_output_counts_rows() -> None:
    summary = summarize_sheet_sync_output(
        "[INFO] row=2 style=音声リール input=https://example.com\n"
        "[DONE] row=2 status=完了 mode=asr segments=3\n"
        "[INFO] row=3 style=画像リール input=https://example.com\n"
        "[ERROR] row=3: bad request\n"
    )

    assert summary == {"rows": 2, "done": 1, "errors": 1}


def test_read_text_file_best_effort_supports_cp932(tmp_path: Path) -> None:
    log_path = tmp_path / "sheet_sync_latest.log"
    log_path.write_bytes("情報: テスト".encode("cp932"))

    assert read_text_file_best_effort(log_path) == "情報: テスト"
