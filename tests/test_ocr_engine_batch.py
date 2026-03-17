from app.ocr_engine import OcrCandidate, OcrEngine, OcrView


def test_extract_text_batch_batches_secondary_fast_rescue():
    engine = OcrEngine.__new__(OcrEngine)
    batched_calls = []

    def build_primary_view(frame):
        return OcrView(image=("primary", frame), x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=0)

    def build_secondary_fast_view(frame):
        return OcrView(image=("secondary", frame), x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=1)

    def readtext_batched_fast(images, *, canvas_size=1600, mag_ratio=1.1, chunk_size=4):
        batched_calls.append((canvas_size, mag_ratio, list(images)))
        results = []
        for kind, frame in images:
            if kind == "primary" and frame == "needs_rescue":
                results.append([('weak text', 0.61)])
            elif kind == "primary":
                results.append([('strong text', 0.91)])
            else:
                results.append([('rescued text', 0.88)])
        return results

    def extract_candidates_from_results(raw_results, view):
        return [
            OcrCandidate(
                text=text,
                confidence=confidence,
                x_center=0.5,
                y_center=0.5,
                variant_rank=view.rank,
            )
            for text, confidence in raw_results
        ]

    def needs_fast_rescue(result):
        return result[0] == 'weak text'

    def needs_fallback(result):
        return False

    engine._build_primary_view = build_primary_view
    engine._build_secondary_fast_view = build_secondary_fast_view
    engine._readtext_batched_fast = readtext_batched_fast
    engine._extract_candidates_from_results = extract_candidates_from_results
    engine._needs_fast_rescue = needs_fast_rescue
    engine._needs_fallback = needs_fallback
    engine.extract_text = lambda frame: ('fallback', 0.5)

    results = engine.extract_text_batch(['needs_rescue', 'already_ok'])

    assert 'rescued text' in results[0][0]
    assert results[1] == ('strong text', 0.91)
    assert batched_calls == [
        (1600, 1.1, [('primary', 'needs_rescue'), ('primary', 'already_ok')]),
        (1920, 1.2, [('secondary', 'needs_rescue')]),
    ]
