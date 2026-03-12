from app.commit_messages import (
    build_auto_summary,
    format_commit_subject,
    rewrite_commit_message_text,
)


def test_build_auto_summary_groups_known_paths():
    summary = build_auto_summary(
        [
            "app/ocr_engine.py",
            "app/frame_sampler.py",
            "tests/test_ocr_engine.py",
        ]
    )

    assert summary == "OCR・セグメント判定・テスト更新"


def test_build_auto_summary_limits_number_of_categories():
    summary = build_auto_summary(
        [
            "app/ocr_engine.py",
            "app/frame_sampler.py",
            "app/pipeline.py",
            "web_ui.py",
            "README.md",
        ]
    )

    assert summary == "OCR・セグメント判定・パイプライン・他更新"


def test_format_commit_subject_adds_date_and_auto_summary():
    subject = format_commit_subject(
        existing_subject="しきい値を調整",
        auto_summary="OCR・セグメント判定更新",
        commit_date="2026-03-12",
    )

    assert subject == "2026-03-12 OCR・セグメント判定更新 | しきい値を調整"


def test_format_commit_subject_keeps_existing_dated_subject():
    subject = format_commit_subject(
        existing_subject="2026-03-12 OCR更新",
        auto_summary="OCR更新",
        commit_date="2026-03-13",
    )

    assert subject == "2026-03-12 OCR更新"


def test_rewrite_commit_message_text_preserves_body():
    updated = rewrite_commit_message_text(
        original_text="細かい補足\n\n本文のメモ\n",
        auto_summary="Web UI更新",
        commit_date="2026-03-12",
    )

    assert updated == "2026-03-12 Web UI更新 | 細かい補足\n\n本文のメモ\n"
