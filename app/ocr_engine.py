from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

import cv2

from .runtime_paths import configure_paddle_runtime_env
from .utils import normalize_text

SUPPORTED_OCR_BACKENDS = ("auto", "paddle", "easyocr")


@dataclass
class OcrView:
    image: object
    x_offset: float
    y_offset: float
    x_scale: float
    y_scale: float
    rank: int


@dataclass
class OcrFragment:
    text: str
    confidence: float
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class OcrCandidate:
    text: str
    confidence: float
    x_center: float
    y_center: float
    variant_rank: int


class OcrEngine:
    def __init__(
        self,
        languages: Iterable[str] = ("ja", "en"),
        gpu: bool = False,
        min_confidence: float = 0.15,
        backend: str = "auto",
    ) -> None:
        self.languages = [token.strip().lower() for token in languages if token.strip()]
        self.min_confidence = min_confidence
        self.gpu = gpu
        self.backend = self._normalize_backend(backend)
        self._backend_name = ""

        if self.backend == "auto":
            errors: list[tuple[str, Exception]] = []
            for backend_name in ("paddle", "easyocr"):
                try:
                    self._initialize_backend(backend_name)
                    return
                except Exception as exc:  # pragma: no cover - fallback path is environment-specific
                    errors.append((backend_name, exc))
            details = " ".join(f"{name} error: {exc!s}" for name, exc in errors)
            raise RuntimeError(f"OCR backend could not be initialized. {details}".strip()) from errors[-1][1]

        try:
            self._initialize_backend(self.backend)
        except Exception as exc:  # pragma: no cover - backend-specific path is environment-specific
            raise RuntimeError(
                f"Requested OCR backend '{self.backend}' could not be initialized: {exc!s}"
            ) from exc

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = (backend or "auto").strip().lower()
        if normalized not in SUPPORTED_OCR_BACKENDS:
            supported = ", ".join(SUPPORTED_OCR_BACKENDS)
            raise ValueError(f"Unsupported OCR backend '{backend}'. Supported values: {supported}")
        return normalized

    def _initialize_backend(self, backend_name: str) -> None:
        if backend_name == "paddle":
            self._init_paddle_backend()
        elif backend_name == "easyocr":
            self._init_easyocr_backend()
        else:  # pragma: no cover - validated by _normalize_backend
            raise ValueError(f"Unsupported OCR backend '{backend_name}'")
        self._backend_name = backend_name

    def _init_paddle_backend(self) -> None:
        configure_paddle_runtime_env()

        from paddleocr import PaddleOCR

        det_model_name, rec_model_name = self._resolve_paddle_model_names()
        self.paddle_reader = PaddleOCR(
            text_detection_model_name=det_model_name,
            text_recognition_model_name=rec_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            enable_hpi=False,
            cpu_threads=4,
            text_rec_score_thresh=0.0,
        )

    def _init_easyocr_backend(self) -> None:
        import easyocr
        import torch

        # EasyOCR on Windows can overwhelm the machine when torch uses all CPU threads.
        try:
            torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.reader = easyocr.Reader(list(self.languages or ("ja", "en")), gpu=self.gpu, verbose=False)

    def _resolve_paddle_model_names(self) -> tuple[str, str]:
        has_japanese = "ja" in self.languages or "japan" in self.languages
        if has_japanese:
            return "PP-OCRv5_mobile_det", "japan_PP-OCRv3_mobile_rec"
        return "PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"

    @staticmethod
    def _grayscale(frame, scale: float = 2.25):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return gray

    @staticmethod
    def _enhance_gray(gray):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (0, 0), 1.2)
        sharpened = cv2.addWeighted(clahe, 1.55, blurred, -0.55, 0)
        return sharpened

    @staticmethod
    def _adaptive_binary(gray):
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )

    @staticmethod
    def _crop(frame, left: float, top: float, right: float, bottom: float):
        height, width = frame.shape[:2]
        x0 = int(width * left)
        y0 = int(height * top)
        x1 = int(width * right)
        y1 = int(height * bottom)
        return frame[y0:y1, x0:x1], left, top, right - left, bottom - top

    def _build_views(self, frame) -> list[OcrView]:
        full_gray = self._grayscale(frame, scale=3.0)
        full_enhanced = self._enhance_gray(full_gray)
        full_binary = self._adaptive_binary(full_enhanced)
        full_binary_inverted = cv2.bitwise_not(full_binary)

        center_crop, cx, cy, cw, ch = self._crop(frame, 0.03, 0.04, 0.97, 0.96)
        center_gray = self._grayscale(center_crop, scale=2.8)
        center_enhanced = self._enhance_gray(center_gray)
        center_binary = self._adaptive_binary(center_enhanced)
        center_binary_inverted = cv2.bitwise_not(center_binary)

        top_crop, tx, ty, tw, th = self._crop(frame, 0.02, 0.02, 0.98, 0.58)
        top_gray = self._grayscale(top_crop, scale=2.6)
        top_enhanced = self._enhance_gray(top_gray)
        top_binary_inverted = cv2.bitwise_not(self._adaptive_binary(top_enhanced))

        bottom_crop, bx, by, bw, bh = self._crop(frame, 0.02, 0.34, 0.98, 0.98)
        bottom_gray = self._grayscale(bottom_crop, scale=2.6)
        bottom_enhanced = self._enhance_gray(bottom_gray)
        bottom_binary_inverted = cv2.bitwise_not(self._adaptive_binary(bottom_enhanced))

        return [
            OcrView(image=full_enhanced, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=0),
            OcrView(image=full_binary, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=1),
            OcrView(image=full_binary_inverted, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=2),
            OcrView(image=frame, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=3),
            OcrView(image=center_enhanced, x_offset=cx, y_offset=cy, x_scale=cw, y_scale=ch, rank=4),
            OcrView(image=center_binary, x_offset=cx, y_offset=cy, x_scale=cw, y_scale=ch, rank=5),
            OcrView(image=center_binary_inverted, x_offset=cx, y_offset=cy, x_scale=cw, y_scale=ch, rank=6),
            OcrView(image=top_enhanced, x_offset=tx, y_offset=ty, x_scale=tw, y_scale=th, rank=7),
            OcrView(image=top_binary_inverted, x_offset=tx, y_offset=ty, x_scale=tw, y_scale=th, rank=8),
            OcrView(image=bottom_enhanced, x_offset=bx, y_offset=by, x_scale=bw, y_scale=bh, rank=9),
            OcrView(image=bottom_binary_inverted, x_offset=bx, y_offset=by, x_scale=bw, y_scale=bh, rank=10),
        ]

    def _build_primary_view(self, frame) -> OcrView:
        center_crop, cx, cy, cw, ch = self._crop(frame, 0.03, 0.03, 0.97, 0.985)
        center_gray = self._grayscale(center_crop, scale=1.6)
        center_enhanced = self._enhance_gray(center_gray)
        return OcrView(image=center_enhanced, x_offset=cx, y_offset=cy, x_scale=cw, y_scale=ch, rank=0)

    def _build_secondary_fast_view(self, frame) -> OcrView:
        gray = self._grayscale(frame, scale=1.45)
        enhanced = self._enhance_gray(gray)
        return OcrView(image=enhanced, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=1)

    def _build_paddle_primary_view(self, frame) -> OcrView:
        crop, cx, cy, cw, ch = self._crop(frame, 0.02, 0.02, 0.98, 0.985)
        return OcrView(image=crop, x_offset=cx, y_offset=cy, x_scale=cw, y_scale=ch, rank=0)

    def _build_paddle_secondary_view(self, frame) -> OcrView:
        return OcrView(image=frame, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=1)

    def _readtext(self, image):
        return self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            beamWidth=12,
            contrast_ths=0.03,
            adjust_contrast=0.75,
            text_threshold=0.55,
            low_text=0.2,
            link_threshold=0.25,
            canvas_size=3200,
            mag_ratio=1.8,
        )

    def _readtext_fast(self, image):
        return self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            decoder="greedy",
            beamWidth=1,
            contrast_ths=0.05,
            adjust_contrast=0.6,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.3,
            canvas_size=1920,
            mag_ratio=1.2,
        )

    def _readtext_batched_fast(
        self,
        images: list[object],
        *,
        canvas_size: int = 1600,
        mag_ratio: float = 1.1,
        chunk_size: int = 4,
    ):
        results = []
        for start in range(0, len(images), chunk_size):
            chunk = images[start : start + chunk_size]
            results.extend(
                self.reader.readtext_batched(
                    chunk,
                    decoder="greedy",
                    beamWidth=1,
                    batch_size=len(chunk),
                    workers=0,
                    detail=1,
                    paragraph=False,
                    contrast_ths=0.05,
                    adjust_contrast=0.6,
                    text_threshold=0.6,
                    low_text=0.3,
                    link_threshold=0.3,
                    canvas_size=canvas_size,
                    mag_ratio=mag_ratio,
                )
            )
        return results

    def _extract_from_view(self, view: OcrView) -> list[OcrCandidate]:
        raw_results = self._readtext(view.image)
        return self._extract_candidates_from_results(raw_results=raw_results, view=view)

    def _extract_from_view_fast(self, view: OcrView) -> list[OcrCandidate]:
        raw_results = self._readtext_fast(view.image)
        return self._extract_candidates_from_results(raw_results=raw_results, view=view)

    def _extract_candidates_from_results(self, raw_results, view: OcrView) -> list[OcrCandidate]:
        image_height, image_width = view.image.shape[:2]
        fragments: list[OcrFragment] = []

        for bbox, raw_text, confidence in raw_results:
            text = normalize_text(raw_text)
            conf = float(confidence)
            if not text or conf < self.min_confidence:
                continue
            if len(text) == 1 and conf < max(self.min_confidence, 0.35):
                continue
            if re.fullmatch(r"[\W_]+", text):
                continue

            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            x0 = view.x_offset + (min(xs) / image_width) * view.x_scale
            x1 = view.x_offset + (max(xs) / image_width) * view.x_scale
            y0 = view.y_offset + (min(ys) / image_height) * view.y_scale
            y1 = view.y_offset + (max(ys) / image_height) * view.y_scale
            fragments.append(
                OcrFragment(
                    text=text,
                    confidence=conf,
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                )
            )

        return self._merge_fragments_into_lines(fragments=fragments, variant_rank=view.rank)

    def _predict_paddle_batch(self, images: list[object]) -> list[object]:
        if not images:
            return []
        return list(self.paddle_reader.predict(images))

    def _extract_paddle_candidates(self, raw_result: object, view: OcrView) -> list[OcrCandidate]:
        if not raw_result:
            return []

        image_height, image_width = view.image.shape[:2]
        texts = raw_result.get("rec_texts") or []
        scores = raw_result.get("rec_scores") or []
        polygons = raw_result.get("rec_polys") or raw_result.get("dt_polys") or []

        candidates: list[OcrCandidate] = []
        for raw_text, raw_score, polygon in zip(texts, scores, polygons):
            text = normalize_text(raw_text)
            confidence = float(raw_score)
            if not text or confidence < self.min_confidence:
                continue
            points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            x_center = view.x_offset + (((min(xs) + max(xs)) / 2.0) / image_width) * view.x_scale
            y_center = view.y_offset + (((min(ys) + max(ys)) / 2.0) / image_height) * view.y_scale
            candidates.append(
                OcrCandidate(
                    text=text,
                    confidence=confidence,
                    x_center=x_center,
                    y_center=y_center,
                    variant_rank=view.rank,
                )
            )
        return candidates

    def _extract_text_primary_batch_paddle(self, frames: list[object]) -> list[tuple[str, float | None]]:
        if not frames:
            return []

        views = [self._build_paddle_primary_view(frame) for frame in frames]
        raw_batches = self._predict_paddle_batch([view.image for view in views])
        return [
            self._finalize_candidates(self._extract_paddle_candidates(raw_result=raw_result, view=view))
            for view, raw_result in zip(views, raw_batches)
        ]

    def _extract_text_batch_paddle(self, frames: list[object]) -> list[tuple[str, float | None]]:
        if not frames:
            return []

        primary_views = [self._build_paddle_primary_view(frame) for frame in frames]
        primary_batches = self._predict_paddle_batch([view.image for view in primary_views])

        candidates_by_index: list[list[OcrCandidate]] = []
        primary_results: list[tuple[str, float | None]] = []
        rescue_indices: list[int] = []

        for index, (view, raw_result) in enumerate(zip(primary_views, primary_batches)):
            candidates = self._extract_paddle_candidates(raw_result=raw_result, view=view)
            primary_result = self._finalize_candidates(candidates)
            candidates_by_index.append(candidates)
            primary_results.append(primary_result)
            if self._needs_fast_rescue(primary_result):
                rescue_indices.append(index)

        if rescue_indices:
            rescue_views = [self._build_paddle_secondary_view(frames[index]) for index in rescue_indices]
            rescue_batches = self._predict_paddle_batch([view.image for view in rescue_views])
            for frame_index, view, raw_result in zip(rescue_indices, rescue_views, rescue_batches):
                candidates_by_index[frame_index].extend(
                    self._extract_paddle_candidates(raw_result=raw_result, view=view)
                )
                primary_results[frame_index] = self._finalize_candidates(candidates_by_index[frame_index])

        return primary_results

    def _extract_text_primary_batch_easyocr(self, frames: list[object]) -> list[tuple[str, float | None]]:
        results: list[tuple[str, float | None]] = []
        for frame in frames:
            primary_view = self._build_primary_view(frame)
            raw_results = self._readtext_fast(primary_view.image)
            candidates = self._extract_candidates_from_results(raw_results=raw_results, view=primary_view)
            results.append(self._finalize_candidates(candidates))
        return results

    def _extract_text_batch_easyocr(self, frames: list[object]) -> list[tuple[str, float | None]]:
        results: list[tuple[str, float | None]] = []
        for frame in frames:
            primary_view = self._build_primary_view(frame)
            raw_results = self._readtext_fast(primary_view.image)
            candidates = self._extract_candidates_from_results(raw_results=raw_results, view=primary_view)
            fast_result = self._finalize_candidates(candidates)

            if self._needs_fast_rescue(fast_result):
                secondary_view = self._build_secondary_fast_view(frame)
                secondary_results = self._readtext_fast(secondary_view.image)
                candidates.extend(
                    self._extract_candidates_from_results(raw_results=secondary_results, view=secondary_view)
                )
                fast_result = self._finalize_candidates(candidates)

            if self._needs_fallback(fast_result):
                results.append(self.extract_text(frame))
            else:
                results.append(fast_result)
        return results

    def extract_text(self, frame) -> tuple[str, float | None]:
        if getattr(self, "_backend_name", "") == "paddle":
            return self._extract_text_batch_paddle([frame])[0]

        candidates: list[OcrCandidate] = []
        for view in self._build_views(frame):
            candidates.extend(self._extract_from_view(view))

        return self._finalize_candidates(candidates)

    def extract_text_primary_batch(self, frames: list[object]) -> list[tuple[str, float | None]]:
        if getattr(self, "_backend_name", "") == "paddle":
            return self._extract_text_primary_batch_paddle(frames)
        if getattr(self, "_backend_name", "") == "easyocr":
            return self._extract_text_primary_batch_easyocr(frames)

        if not frames:
            return []

        primary_views = [self._build_primary_view(frame) for frame in frames]
        raw_batches = self._readtext_batched_fast(
            [view.image for view in primary_views],
            canvas_size=1600,
            mag_ratio=1.1,
        )

        results: list[tuple[str, float | None]] = []
        for view, raw_results in zip(primary_views, raw_batches):
            candidates = self._extract_candidates_from_results(raw_results=raw_results, view=view)
            results.append(self._finalize_candidates(candidates))
        return results

    def extract_text_batch(self, frames: list[object]) -> list[tuple[str, float | None]]:
        if getattr(self, "_backend_name", "") == "paddle":
            return self._extract_text_batch_paddle(frames)
        if getattr(self, "_backend_name", "") == "easyocr":
            return self._extract_text_batch_easyocr(frames)

        if not frames:
            return []

        primary_views = [self._build_primary_view(frame) for frame in frames]
        raw_batches = self._readtext_batched_fast(
            [view.image for view in primary_views],
            canvas_size=1600,
            mag_ratio=1.1,
        )

        candidates_by_index: list[list[OcrCandidate]] = []
        fast_results: list[tuple[str, float | None]] = []
        rescue_indices: list[int] = []

        for index, (view, raw_results) in enumerate(zip(primary_views, raw_batches)):
            candidates = self._extract_candidates_from_results(raw_results=raw_results, view=view)
            fast_result = self._finalize_candidates(candidates)
            candidates_by_index.append(candidates)
            fast_results.append(fast_result)
            if self._needs_fast_rescue(fast_result):
                rescue_indices.append(index)

        if rescue_indices:
            secondary_views = [self._build_secondary_fast_view(frames[index]) for index in rescue_indices]
            secondary_batches = self._readtext_batched_fast(
                [view.image for view in secondary_views],
                canvas_size=1920,
                mag_ratio=1.2,
            )
            for frame_index, view, raw_results in zip(rescue_indices, secondary_views, secondary_batches):
                candidates = candidates_by_index[frame_index]
                candidates.extend(self._extract_candidates_from_results(raw_results=raw_results, view=view))
                fast_results[frame_index] = self._finalize_candidates(candidates)

        results: list[tuple[str, float | None]] = []
        for frame, fast_result in zip(frames, fast_results):
            if self._needs_fallback(fast_result):
                results.append(self.extract_text(frame))
            else:
                results.append(fast_result)
        return results

    @staticmethod
    def _finalize_candidates(candidates: list[OcrCandidate]) -> tuple[str, float | None]:
        merged = OcrEngine._merge_candidates(candidates)
        if not merged:
            return "", None

        blocks = OcrEngine._build_output_blocks(merged)
        merged_text = OcrEngine._render_output_blocks(blocks)
        avg_confidence = sum(candidate.confidence for candidate in merged) / len(merged)
        if not merged_text:
            return "", None
        return merged_text, round(avg_confidence, 4)

    @staticmethod
    def _needs_fallback(result: tuple[str, float | None]) -> bool:
        text, confidence = result
        normalized = normalize_text(text)
        line_count = len([line for line in text.splitlines() if normalize_text(line)])
        return (
            not normalized
            or confidence is None
            or confidence < 0.5
            or len(normalized) < 6
            or (line_count <= 1 and len(normalized) < 18)
        )

    @staticmethod
    def _needs_fast_rescue(result: tuple[str, float | None]) -> bool:
        text, confidence = result
        normalized = normalize_text(text)
        line_count = len([line for line in text.splitlines() if normalize_text(line)])
        return (
            not normalized
            or confidence is None
            or confidence < 0.72
            or line_count < 3
        )

    @classmethod
    def _build_output_blocks(cls, candidates: list[OcrCandidate]) -> list[list[OcrCandidate]]:
        ordered = sorted(candidates, key=lambda item: (item.y_center, item.x_center))
        two_column_blocks = cls._split_two_column_blocks(ordered)
        if two_column_blocks is not None:
            return two_column_blocks
        return [ordered]

    @classmethod
    def _split_two_column_blocks(
        cls,
        ordered: list[OcrCandidate],
    ) -> list[list[OcrCandidate]] | None:
        if len(ordered) < 7:
            return None

        y_values = [candidate.y_center for candidate in ordered]
        min_y = min(y_values)
        max_y = max(y_values)
        y_span = max_y - min_y
        if y_span < 0.18:
            return None

        top_cutoff = min_y + max(0.06, y_span * 0.12)
        bottom_cutoff = max_y - max(0.08, y_span * 0.16)
        centered = [candidate for candidate in ordered if 0.36 <= candidate.x_center <= 0.64]
        title = [candidate for candidate in centered if candidate.y_center <= top_cutoff]
        footer = [candidate for candidate in centered if candidate.y_center >= bottom_cutoff]

        reserved_ids = {id(candidate) for candidate in title + footer}
        body = [candidate for candidate in ordered if id(candidate) not in reserved_ids]
        left = [candidate for candidate in body if candidate.x_center < 0.5]
        right = [candidate for candidate in body if candidate.x_center >= 0.5]

        if not cls._looks_like_two_column_layout(left=left, right=right):
            return None

        blocks = [title, left, right, footer]
        return [
            sorted(block, key=lambda item: (item.y_center, item.x_center))
            for block in blocks
            if block
        ]

    @staticmethod
    def _looks_like_two_column_layout(
        left: list[OcrCandidate],
        right: list[OcrCandidate],
    ) -> bool:
        if len(left) < 3 or len(right) < 3:
            return False

        left_median_x = OcrEngine._median(candidate.x_center for candidate in left)
        right_median_x = OcrEngine._median(candidate.x_center for candidate in right)
        if left_median_x >= 0.44 or right_median_x <= 0.56:
            return False

        left_y = [candidate.y_center for candidate in left]
        right_y = [candidate.y_center for candidate in right]
        left_span = max(left_y) - min(left_y)
        right_span = max(right_y) - min(right_y)
        if left_span <= 0.05 or right_span <= 0.05:
            return False

        overlap = max(0.0, min(max(left_y), max(right_y)) - max(min(left_y), min(right_y)))
        overlap_ratio = overlap / max(0.001, min(left_span, right_span))
        return overlap_ratio >= 0.45

    @staticmethod
    def _median(values: Iterable[float]) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        middle = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    @classmethod
    def _render_output_blocks(cls, blocks: list[list[OcrCandidate]]) -> str:
        rendered_lines: list[str] = []
        for block in blocks:
            block_lines = [
                cleaned
                for candidate in block
                if cls._should_keep_output_candidate(candidate)
                for cleaned in [cls._clean_output_line(candidate.text)]
                if cleaned
            ]
            block_lines = cls._dedupe_output_lines(block_lines)
            if not block_lines:
                continue
            if rendered_lines:
                rendered_lines.append("")
            rendered_lines.extend(block_lines)

        return "\n".join(cls._dedupe_output_lines(rendered_lines))

    @staticmethod
    def _clean_output_line(text: str) -> str:
        cleaned = normalize_text(text)
        cleaned = re.sub(r"^[・•●▪◦·･]+", "", cleaned)
        cleaned = re.sub(r"^[\"'`´‘’]+(?=\S)", "", cleaned)
        cleaned = re.sub(r"^\.\s*(?=[^0-9])", "", cleaned)
        return normalize_text(cleaned)

    @staticmethod
    def _should_keep_output_candidate(candidate: OcrCandidate) -> bool:
        cleaned = OcrEngine._clean_output_line(candidate.text)
        if not cleaned:
            return False

        compact = cleaned.replace(" ", "")
        ascii_count = sum(char.isascii() for char in compact)
        digit_count = sum(char.isdigit() for char in compact)
        alpha_count = sum(char.isalpha() for char in compact)
        noise_count = sum(char in "|[]{}<>~`-_/" for char in compact)
        ascii_ratio = ascii_count / max(len(compact), 1)

        if len(compact) == 1 and candidate.confidence < 0.9:
            return False
        if digit_count == len(compact) and len(compact) <= 2 and candidate.confidence < 0.98:
            return False
        if candidate.confidence < 0.45 and len(compact) < 5:
            return False
        if ascii_ratio >= 0.7 and len(compact) <= 6 and candidate.confidence < 0.9:
            return False
        if noise_count >= 2 and len(compact) <= 8:
            return False
        if alpha_count >= 2 and ascii_ratio >= 0.5 and len(compact) <= 4 and candidate.confidence < 0.92:
            return False
        return True

    @staticmethod
    def _dedupe_output_lines(lines: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()

        for line in lines:
            normalized = normalize_text(line)
            if not normalized:
                if deduped and deduped[-1] != "":
                    deduped.append("")
                continue

            if deduped and deduped[-1] and normalize_text(deduped[-1]) == normalized:
                continue
            if len(normalized) >= 3 and normalized in seen:
                continue

            deduped.append(normalized)
            if len(normalized) >= 3:
                seen.add(normalized)

        while deduped and deduped[0] == "":
            deduped.pop(0)
        while deduped and deduped[-1] == "":
            deduped.pop()
        return deduped

    @classmethod
    def _merge_fragments_into_lines(
        cls,
        fragments: list[OcrFragment],
        variant_rank: int,
    ) -> list[OcrCandidate]:
        if not fragments:
            return []

        ordered = sorted(fragments, key=lambda item: (item.y_center, item.x0))
        lines: list[dict] = []

        for fragment in ordered:
            if lines and cls._can_append_to_line(lines[-1], fragment):
                separator = cls._join_separator(lines[-1]["text"], fragment.text)
                lines[-1]["text"] = normalize_text(lines[-1]["text"] + separator + fragment.text)
                lines[-1]["x1"] = max(lines[-1]["x1"], fragment.x1)
                lines[-1]["y0"] = min(lines[-1]["y0"], fragment.y0)
                lines[-1]["y1"] = max(lines[-1]["y1"], fragment.y1)
                lines[-1]["confidence_values"].append(fragment.confidence)
            else:
                lines.append(
                    {
                        "text": fragment.text,
                        "x0": fragment.x0,
                        "x1": fragment.x1,
                        "y0": fragment.y0,
                        "y1": fragment.y1,
                        "confidence_values": [fragment.confidence],
                    }
                )

        candidates: list[OcrCandidate] = []
        for line in lines:
            candidates.append(
                OcrCandidate(
                    text=line["text"],
                    confidence=sum(line["confidence_values"]) / len(line["confidence_values"]),
                    x_center=(line["x0"] + line["x1"]) / 2.0,
                    y_center=(line["y0"] + line["y1"]) / 2.0,
                    variant_rank=variant_rank,
                )
            )
        return candidates

    @staticmethod
    def _can_append_to_line(line: dict, fragment: OcrFragment) -> bool:
        same_row = abs(((line["y0"] + line["y1"]) / 2.0) - fragment.y_center) <= 0.045
        close_gap = fragment.x0 - line["x1"] <= 0.08
        return same_row and close_gap

    @staticmethod
    def _join_separator(left: str, right: str) -> str:
        if not left or not right:
            return ""
        if OcrEngine._needs_space(left[-1], right[0]):
            return " "
        return ""

    @staticmethod
    def _needs_space(left_char: str, right_char: str) -> bool:
        return left_char.isascii() and right_char.isascii() and left_char.isalnum() and right_char.isalnum()

    @classmethod
    def _merge_candidates(cls, candidates: list[OcrCandidate]) -> list[OcrCandidate]:
        if not candidates:
            return []

        ordered = sorted(candidates, key=cls._candidate_score, reverse=True)
        groups: list[list[OcrCandidate]] = []

        for candidate in ordered:
            matched_group: list[OcrCandidate] | None = None
            for group in groups:
                if cls._same_candidate_group(candidate, group[0]):
                    matched_group = group
                    break
            if matched_group is None:
                groups.append([candidate])
            else:
                matched_group.append(candidate)
                matched_group.sort(key=cls._candidate_score, reverse=True)

        return [group[0] for group in groups]

    @staticmethod
    def _candidate_score(candidate: OcrCandidate) -> tuple[float, int, float]:
        return (candidate.confidence, len(candidate.text), -candidate.variant_rank)

    @staticmethod
    def _same_candidate_group(left: OcrCandidate, right: OcrCandidate) -> bool:
        y_close = abs(left.y_center - right.y_center) <= 0.06
        x_close = abs(left.x_center - right.x_center) <= 0.35
        similar = SequenceMatcher(None, left.text, right.text).ratio() >= 0.58
        contained = left.text in right.text or right.text in left.text
        return y_close and (x_close or contained) and (similar or contained)
