from pathlib import Path

from app.sheets_dashboard import (
    MANAGED_SHEET_SYNC_SPREADSHEETS,
    RUN_MODE_RERUN_IMAGES,
    RUN_MODE_RETRY_ERRORS,
    RUN_MODE_TEST_ONE,
    SheetSyncPreset,
    SheetSyncRunRequest,
    get_sheet_sync_save_status,
    get_sheet_sync_spreadsheet_label,
    group_sheet_sync_presets,
    has_sheet_sync_preset_content,
    build_sheet_sync_command,
    load_sheet_sync_presets,
    read_text_file_best_effort,
    resolve_sheet_sync_selected_preset,
    save_sheet_sync_presets,
    summarize_sheet_sync_output,
    upsert_sheet_sync_preset,
)


def test_load_sheet_sync_presets_returns_default_when_missing(tmp_path: Path) -> None:
    presets, selected_name = load_sheet_sync_presets(tmp_path / "missing.json")

    assert selected_name == "現在の運用設定"
    assert any(preset.name == "現在の運用設定" for preset in presets)
    assert {preset.spreadsheet_id for preset in presets} >= {
        managed["spreadsheet_id"] for managed in MANAGED_SHEET_SYNC_SPREADSHEETS
    }


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

    assert selected_name == "B"
    assert loaded_presets[:2] == [
        SheetSyncPreset(
            name="A",
            service_account_file=r"C:\secure\a.json",
            spreadsheet_id="sheet-a",
            sheet_name="投稿データA",
            url_column="G",
            spreadsheet_label="A",
        ),
        SheetSyncPreset(
            name="B",
            service_account_file=r"C:\secure\b.json",
            spreadsheet_id="sheet-b",
            sheet_name="投稿データB",
            url_column="H",
            spreadsheet_label="B",
        ),
    ]
    assert {preset.spreadsheet_id for preset in loaded_presets} >= {
        "sheet-a",
        "sheet-b",
        *(managed["spreadsheet_id"] for managed in MANAGED_SHEET_SYNC_SPREADSHEETS),
    }


def test_group_sheet_sync_presets_uses_spreadsheet_label() -> None:
    grouped = group_sheet_sync_presets(
        [
            SheetSyncPreset(
                name="A",
                service_account_file="a.json",
                spreadsheet_id="sheet-a",
                sheet_name="tab-a",
                spreadsheet_label="Instagram",
            ),
            SheetSyncPreset(
                name="B",
                service_account_file="b.json",
                spreadsheet_id="sheet-a",
                sheet_name="tab-b",
                spreadsheet_label="Instagram",
            ),
            SheetSyncPreset(
                name="C",
                service_account_file="c.json",
                spreadsheet_id="sheet-c",
                sheet_name="tab-c",
                spreadsheet_label="Shorts",
            ),
        ]
    )

    assert list(grouped.keys()) == ["Instagram", "Shorts"]
    assert [preset.name for preset in grouped["Instagram"]] == ["A", "B"]


def test_get_sheet_sync_spreadsheet_label_uses_known_mapping() -> None:
    preset = SheetSyncPreset(
        name="Custom",
        service_account_file="svc.json",
        spreadsheet_id="11qZWfM1fB-tY5KRUBmK0YJXX_xyUGU32-XgGI07oS0k",
        sheet_name="",
    )

    assert get_sheet_sync_spreadsheet_label(preset) == "インスタ運用シート"


def test_has_sheet_sync_preset_content_detects_blank_and_non_blank_values() -> None:
    assert not has_sheet_sync_preset_content(
        SheetSyncPreset(name="", service_account_file="", spreadsheet_id="", sheet_name="")
    )
    assert has_sheet_sync_preset_content(
        SheetSyncPreset(name="", service_account_file="", spreadsheet_id="", sheet_name="投稿データ260301")
    )


def test_get_sheet_sync_save_status_reports_saved_and_unsaved_states() -> None:
    saved = SheetSyncPreset(
        name="260301 本番",
        spreadsheet_label="インスタ運用シート",
        service_account_file="svc.json",
        spreadsheet_id="sheet-a",
        sheet_name="投稿データ260301",
    )

    assert get_sheet_sync_save_status(saved, selected_name="260301 本番", saved_preset=saved) == (
        "success",
        "保存済み: インスタ運用シート / 260301 本番",
    )
    assert get_sheet_sync_save_status(
        SheetSyncPreset(
            name="260301 本番",
            spreadsheet_label="インスタ運用シート",
            service_account_file="svc.json",
            spreadsheet_id="sheet-a",
            sheet_name="投稿データ260302",
        ),
        selected_name="260301 本番",
        saved_preset=saved,
    ) == (
        "warning",
        "未保存の変更があります。「設定を保存」で反映してください。",
    )
    assert get_sheet_sync_save_status(
        SheetSyncPreset(
            name="新規",
            spreadsheet_label="インスタ運用シート",
            service_account_file="svc.json",
            spreadsheet_id="sheet-a",
            sheet_name="",
        ),
        selected_name="新規設定",
        saved_preset=None,
    ) == (
        "warning",
        "この新規設定はまだ保存されていません。",
    )


def test_upsert_sheet_sync_preset_renames_existing_preset_and_preserves_order() -> None:
    first = SheetSyncPreset(name="A", service_account_file="a.json", spreadsheet_id="sheet-a", sheet_name="tab-a")
    second = SheetSyncPreset(name="B", service_account_file="b.json", spreadsheet_id="sheet-b", sheet_name="tab-b")

    updated = upsert_sheet_sync_preset(
        {"A": first, "B": second},
        SheetSyncPreset(name="Renamed", service_account_file="a.json", spreadsheet_id="sheet-a", sheet_name="tab-a"),
        previous_name="A",
    )

    assert list(updated.keys()) == ["Renamed", "B"]
    assert updated["Renamed"].sheet_name == "tab-a"


def test_upsert_sheet_sync_preset_rejects_duplicate_name() -> None:
    first = SheetSyncPreset(name="A", service_account_file="a.json", spreadsheet_id="sheet-a", sheet_name="tab-a")
    second = SheetSyncPreset(name="B", service_account_file="b.json", spreadsheet_id="sheet-b", sheet_name="tab-b")

    try:
        upsert_sheet_sync_preset(
            {"A": first, "B": second},
            SheetSyncPreset(name="B", service_account_file="a.json", spreadsheet_id="sheet-a", sheet_name="tab-a"),
            previous_name="A",
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected duplicate preset name to raise ValueError")


def test_resolve_sheet_sync_selected_preset_prefers_pending_value() -> None:
    resolved = resolve_sheet_sync_selected_preset(
        ["A", "B", "新規設定"],
        selected_name="A",
        widget_value="B",
        pending_name="新規設定",
    )

    assert resolved == "新規設定"


def test_resolve_sheet_sync_selected_preset_falls_back_to_first_option() -> None:
    resolved = resolve_sheet_sync_selected_preset(
        ["A", "B"],
        selected_name="missing",
        widget_value="",
        pending_name="",
    )

    assert resolved == "A"


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
    assert "--auto-detect-layout" in command
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
    assert "--auto-detect-layout" in test_one
    assert "--auto-detect-layout" in retry


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
