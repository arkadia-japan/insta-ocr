from app.sheets_dashboard import (
    RUN_MODE_RERUN_IMAGES,
    RUN_MODE_RERUN_LOW_QUALITY_OCR,
    RUN_MODE_RETRY_ERRORS,
    RUN_MODE_TEST_ONE,
    SheetSyncPreset,
    SheetSyncRunRequest,
    build_sheet_sync_command,
    resolve_sheet_sync_ocr_backend,
)


def _build_preset() -> SheetSyncPreset:
    return SheetSyncPreset(
        name="preset",
        service_account_file=r"C:\secure\svc.json",
        spreadsheet_id="spreadsheet-id",
        sheet_name="sheet-name",
        url_column="G",
    )


def test_build_sheet_sync_command_includes_requested_ocr_backend():
    command = build_sheet_sync_command(
        "python",
        _build_preset(),
        SheetSyncRunRequest(
            mode=RUN_MODE_RERUN_IMAGES,
            output_dir=r"C:\runtime\output",
            download_dir=r"C:\runtime\downloads",
            ocr_backend="easyocr",
        ),
    )

    assert command[command.index("--ocr-backend") + 1] == "paddle"


def test_build_sheet_sync_command_uses_auto_backend_by_default():
    test_one = build_sheet_sync_command(
        "python",
        _build_preset(),
        SheetSyncRunRequest(
            mode=RUN_MODE_TEST_ONE,
            output_dir=r"C:\runtime\output",
            download_dir=r"C:\runtime\downloads",
        ),
    )
    retry = build_sheet_sync_command(
        "python",
        _build_preset(),
        SheetSyncRunRequest(
            mode=RUN_MODE_RETRY_ERRORS,
            output_dir=r"C:\runtime\output",
            download_dir=r"C:\runtime\downloads",
        ),
    )

    assert test_one[test_one.index("--ocr-backend") + 1] == "auto"
    assert retry[retry.index("--ocr-backend") + 1] == "auto"


def test_build_sheet_sync_command_forces_easyocr_for_low_quality_rerun():
    command = build_sheet_sync_command(
        "python",
        _build_preset(),
        SheetSyncRunRequest(
            mode=RUN_MODE_RERUN_LOW_QUALITY_OCR,
            output_dir=r"C:\runtime\output",
            download_dir=r"C:\runtime\downloads",
            ocr_backend="auto",
        ),
    )

    assert command[command.index("--ocr-backend") + 1] == "easyocr"
    assert "--rerun-low-quality-ocr" in command


def test_resolve_sheet_sync_ocr_backend_uses_easyocr_only_for_single_test():
    assert (
        resolve_sheet_sync_ocr_backend(
            SheetSyncRunRequest(
                mode=RUN_MODE_RERUN_IMAGES,
                output_dir=r"C:\runtime\output",
                download_dir=r"C:\runtime\downloads",
                ocr_backend="easyocr",
            )
        )
        == "paddle"
    )
    assert (
        resolve_sheet_sync_ocr_backend(
            SheetSyncRunRequest(
                mode=RUN_MODE_TEST_ONE,
                output_dir=r"C:\runtime\output",
                download_dir=r"C:\runtime\downloads",
                ocr_backend="easyocr",
            )
        )
        == "easyocr"
    )
    assert (
        resolve_sheet_sync_ocr_backend(
            SheetSyncRunRequest(
                mode=RUN_MODE_RERUN_LOW_QUALITY_OCR,
                output_dir=r"C:\runtime\output",
                download_dir=r"C:\runtime\downloads",
                ocr_backend="auto",
            )
        )
        == "easyocr"
    )
