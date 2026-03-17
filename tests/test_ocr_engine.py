import pytest

from app.ocr_engine import OcrCandidate, OcrEngine, OcrFragment


def test_merge_fragments_into_lines_joins_same_row():
    fragments = [
        OcrFragment(text="恋愛", confidence=0.9, x0=0.10, x1=0.20, y0=0.20, y1=0.25),
        OcrFragment(text="テクニック", confidence=0.88, x0=0.21, x1=0.36, y0=0.205, y1=0.255),
        OcrFragment(text="保存版", confidence=0.84, x0=0.12, x1=0.26, y0=0.35, y1=0.40),
    ]

    lines = OcrEngine._merge_fragments_into_lines(fragments=fragments, variant_rank=0)

    assert len(lines) == 2
    assert lines[0].text == "恋愛テクニック"
    assert lines[1].text == "保存版"


def test_merge_candidates_prefers_high_confidence_text():
    candidates = [
        OcrCandidate(text="モテるLINE", confidence=0.70, x_center=0.4, y_center=0.2, variant_rank=2),
        OcrCandidate(text="モテるLINE", confidence=0.92, x_center=0.41, y_center=0.21, variant_rank=0),
        OcrCandidate(text="距離感がうまい", confidence=0.80, x_center=0.3, y_center=0.4, variant_rank=0),
    ]

    merged = OcrEngine._merge_candidates(candidates)

    assert len(merged) == 2
    assert merged[0].text == "モテるLINE"


def test_finalize_candidates_orders_two_column_layout_and_dedupes_lines():
    candidates = [
        OcrCandidate(text="沼る男の言い方", confidence=0.96, x_center=0.50, y_center=0.08, variant_rank=0),
        OcrCandidate(text="非モテ", confidence=0.93, x_center=0.18, y_center=0.22, variant_rank=0),
        OcrCandidate(text="モテ男", confidence=0.92, x_center=0.77, y_center=0.22, variant_rank=0),
        OcrCandidate(text="・トイレ行ってくる", confidence=0.90, x_center=0.18, y_center=0.31, variant_rank=0),
        OcrCandidate(text="・悪くないね", confidence=0.90, x_center=0.18, y_center=0.38, variant_rank=0),
        OcrCandidate(text="・手つなぐ？", confidence=0.89, x_center=0.18, y_center=0.45, variant_rank=0),
        OcrCandidate(text="・お手洗い行ってくるね", confidence=0.91, x_center=0.78, y_center=0.31, variant_rank=0),
        OcrCandidate(text="・それ似合ってるよ", confidence=0.91, x_center=0.78, y_center=0.38, variant_rank=0),
        OcrCandidate(text="・こっちおいで", confidence=0.90, x_center=0.78, y_center=0.45, variant_rank=0),
        OcrCandidate(
            text="いつでも見返せるようにいいねと保存",
            confidence=0.94,
            x_center=0.50,
            y_center=0.90,
            variant_rank=0,
        ),
        OcrCandidate(
            text="いつでも見返せるようにいいねと保存",
            confidence=0.70,
            x_center=0.52,
            y_center=0.91,
            variant_rank=1,
        ),
    ]

    text, confidence = OcrEngine._finalize_candidates(candidates)

    assert text == (
        "沼る男の言い方\n\n"
        "非モテ\n"
        "トイレ行ってくる\n"
        "悪くないね\n"
        "手つなぐ？\n\n"
        "モテ男\n"
        "お手洗い行ってくるね\n"
        "それ似合ってるよ\n"
        "こっちおいで\n\n"
        "いつでも見返せるようにいいねと保存"
    )
    assert confidence == pytest.approx(0.916, abs=1e-4)


def test_finalize_candidates_dedupes_single_column_output():
    candidates = [
        OcrCandidate(text="タイトル", confidence=0.8, x_center=0.4, y_center=0.1, variant_rank=0),
        OcrCandidate(text="・同じ行", confidence=0.8, x_center=0.4, y_center=0.2, variant_rank=0),
        OcrCandidate(text="同じ行", confidence=0.7, x_center=0.4, y_center=0.3, variant_rank=1),
        OcrCandidate(text="'別の行", confidence=0.8, x_center=0.4, y_center=0.4, variant_rank=0),
        OcrCandidate(text="タイトル", confidence=0.6, x_center=0.4, y_center=0.5, variant_rank=2),
    ]

    text, _ = OcrEngine._finalize_candidates(candidates)

    assert text == "タイトル\n同じ行\n別の行"
