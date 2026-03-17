from app.text_corrections import apply_text_corrections, load_text_corrections


def test_apply_text_corrections_normalizes_cta_line() -> None:
    corrections = load_text_corrections()

    corrected = apply_text_corrections("メ気に入つたらフオロー", corrections)

    assert corrected == "※気に入ったらフォロー"


def test_apply_text_corrections_normalizes_cta_line_in_multiline_text() -> None:
    corrections = load_text_corrections()

    corrected = apply_text_corrections("特徴\n気に入ったらフォ口ー", corrections)

    assert corrected == "特徴\n※気に入ったらフォロー"


def test_apply_text_corrections_keeps_non_cta_line() -> None:
    corrections = load_text_corrections()

    corrected = apply_text_corrections("距離感がうまい男", corrections)

    assert corrected == "距離感がうまい男"


def test_apply_text_corrections_normalizes_common_ocr_typo() -> None:
    corrections = load_text_corrections()

    corrected = apply_text_corrections("ど一も", corrections)

    assert corrected == "どうも"


def test_apply_text_corrections_normalizes_follow_benefit_line() -> None:
    corrections = load_text_corrections()

    corrected = apply_text_corrections("糸フォローで恋愛運上がります。", corrections)

    assert corrected == "※フォローで恋愛運上がります。"


def test_apply_text_corrections_formats_comparison_layout_and_footer() -> None:
    corrections = load_text_corrections()

    corrected = apply_text_corrections(
        (
            "沼る男の言い方\n"
            "非モテ\n"
            "トイレ行ってくる\n"
            "悪くないね\n"
            "モテ男\n"
            "お手洗い行ってくるね\n"
            "俺は00派かな\n"
            "0日空いてる、会おう\n"
            "いつでも見返せるようにいいねと保存\n"
            "糸フォ口一で恋愛運上がります。\n"
            "※フォローで恋愛運上がります。\n"
            "糸フォローで恋愛運上がります"
        ),
        corrections,
    )

    assert corrected == (
        "沼る男の言い方\n\n"
        "非モテ\n"
        "トイレ行ってくる\n"
        "悪くないね\n\n"
        "モテ男\n"
        "お手洗い行ってくるね\n"
        "俺は○○派かな\n"
        "○日空いてる、会おう\n\n"
        "いつでも見返せるようにいいねと保存\n"
        "※フォローで恋愛運上がります。"
    )
