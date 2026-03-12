from app.ocr_engine import OcrCandidate, OcrEngine, OcrFragment


def test_merge_fragments_into_lines_joins_same_row():
    fragments = [
        OcrFragment(text="女に追われる", confidence=0.9, x0=0.10, x1=0.30, y0=0.20, y1=0.25),
        OcrFragment(text="男の特徴", confidence=0.88, x0=0.31, x1=0.48, y0=0.205, y1=0.255),
        OcrFragment(text="保存して", confidence=0.84, x0=0.12, x1=0.26, y0=0.35, y1=0.40),
    ]

    lines = OcrEngine._merge_fragments_into_lines(fragments=fragments, variant_rank=0)

    assert len(lines) == 2
    assert lines[0].text == "女に追われる男の特徴"
    assert lines[1].text == "保存して"


def test_merge_candidates_prefers_high_confidence_text():
    candidates = [
        OcrCandidate(text="男の特微", confidence=0.70, x_center=0.4, y_center=0.2, variant_rank=2),
        OcrCandidate(text="男の特徴", confidence=0.92, x_center=0.41, y_center=0.21, variant_rank=0),
        OcrCandidate(text="保存して", confidence=0.80, x_center=0.3, y_center=0.4, variant_rank=0),
    ]

    merged = OcrEngine._merge_candidates(candidates)

    assert len(merged) == 2
    assert merged[0].text == "男の特徴"
