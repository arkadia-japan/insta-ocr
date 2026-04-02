import pytest

from app.ocr_engine import OcrCandidate, OcrEngine


def test_init_uses_requested_easyocr_backend(monkeypatch):
    calls: list[str] = []

    def init_paddle(self):
        calls.append("paddle")
        raise AssertionError("paddle should not be used")

    def init_easyocr(self):
        calls.append("easyocr")

    monkeypatch.setattr(OcrEngine, "_init_paddle_backend", init_paddle)
    monkeypatch.setattr(OcrEngine, "_init_easyocr_backend", init_easyocr)

    engine = OcrEngine(backend="easyocr")

    assert engine._backend_name == "easyocr"
    assert calls == ["easyocr"]


def test_init_auto_falls_back_to_easyocr(monkeypatch):
    calls: list[str] = []

    def init_paddle(self):
        calls.append("paddle")
        raise RuntimeError("paddle unavailable")

    def init_easyocr(self):
        calls.append("easyocr")

    monkeypatch.setattr(OcrEngine, "_init_paddle_backend", init_paddle)
    monkeypatch.setattr(OcrEngine, "_init_easyocr_backend", init_easyocr)

    engine = OcrEngine(backend="auto")

    assert engine._backend_name == "easyocr"
    assert calls == ["paddle", "easyocr"]


def test_init_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported OCR backend"):
        OcrEngine(backend="unknown")


def test_easyocr_batch_path_runs_frames_sequentially():
    engine = OcrEngine.__new__(OcrEngine)
    engine._backend_name = "easyocr"
    calls: list[tuple[str, str]] = []

    def build_primary_view(frame):
        return type("View", (), {"image": ("primary", frame), "rank": 0})()

    def build_secondary_fast_view(frame):
        return type("View", (), {"image": ("secondary", frame), "rank": 1})()

    def readtext_fast(image):
        kind, frame = image
        calls.append((kind, frame))
        if kind == "primary" and frame == "needs_rescue":
            return [("weak text", 0.61)]
        if kind == "primary":
            return [("strong text", 0.91)]
        return [("rescued text", 0.88)]

    def extract_candidates_from_results(raw_results, view):
        return [type("Candidate", (), {"text": raw_results[0][0], "confidence": raw_results[0][1]})()]

    def finalize_candidates(candidates):
        return candidates[-1].text, candidates[-1].confidence

    def needs_fast_rescue(result):
        return result[0] == "weak text"

    def needs_fallback(result):
        return False

    engine._build_primary_view = build_primary_view
    engine._build_secondary_fast_view = build_secondary_fast_view
    engine._readtext_fast = readtext_fast
    engine._extract_candidates_from_results = extract_candidates_from_results
    engine._finalize_candidates = finalize_candidates
    engine._needs_fast_rescue = needs_fast_rescue
    engine._needs_fallback = needs_fallback
    engine.extract_text = lambda frame: ("fallback", 0.5)

    results = engine.extract_text_batch(["needs_rescue", "already_ok"])

    assert results == [("rescued text", 0.88), ("strong text", 0.91)]
    assert calls == [
        ("primary", "needs_rescue"),
        ("secondary", "needs_rescue"),
        ("primary", "already_ok"),
    ]


def test_render_output_blocks_filters_short_noisy_candidates():
    blocks = [
        [
            OcrCandidate(text="LINEで会話が続く話題", confidence=0.95, x_center=0.5, y_center=0.2, variant_rank=0),
            OcrCandidate(text="1", confidence=0.54, x_center=0.5, y_center=0.3, variant_rank=0),
            OcrCandidate(text="Trt", confidence=0.62, x_center=0.5, y_center=0.4, variant_rank=0),
            OcrCandidate(text="返信しやすい質問をする", confidence=0.91, x_center=0.5, y_center=0.5, variant_rank=0),
        ]
    ]

    rendered = OcrEngine._render_output_blocks(blocks)

    assert rendered == "LINEで会話が続く話題\n返信しやすい質問をする"
