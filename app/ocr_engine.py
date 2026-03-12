from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

import cv2

from .utils import normalize_text


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
    ) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "easyocr is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        self.min_confidence = min_confidence

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
        full_gray = self._grayscale(frame)
        full_enhanced = self._enhance_gray(full_gray)
        full_binary = self._adaptive_binary(full_enhanced)

        center_crop, cx, cy, cw, ch = self._crop(frame, 0.05, 0.06, 0.95, 0.94)
        center_gray = self._grayscale(center_crop)
        center_enhanced = self._enhance_gray(center_gray)
        center_binary = self._adaptive_binary(center_enhanced)

        return [
            OcrView(image=full_enhanced, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=0),
            OcrView(image=full_binary, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=1),
            OcrView(image=frame, x_offset=0.0, y_offset=0.0, x_scale=1.0, y_scale=1.0, rank=2),
            OcrView(image=center_enhanced, x_offset=cx, y_offset=cy, x_scale=cw, y_scale=ch, rank=3),
            OcrView(image=center_binary, x_offset=cx, y_offset=cy, x_scale=cw, y_scale=ch, rank=4),
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

    def _readtext(self, image):
        return self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            beamWidth=10,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.3,
            canvas_size=2560,
            mag_ratio=1.5,
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

    def _readtext_batched_fast(self, images: list[object]):
        results = []
        chunk_size = 2
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
                    canvas_size=1600,
                    mag_ratio=1.1,
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

    def extract_text(self, frame) -> tuple[str, float | None]:
        candidates: list[OcrCandidate] = []
        for view in self._build_views(frame):
            candidates.extend(self._extract_from_view(view))

        merged = self._merge_candidates(candidates)
        if not merged:
            return "", None

        ordered = sorted(merged, key=lambda item: (item.y_center, item.x_center))
        merged_text = "\n".join(candidate.text for candidate in ordered)
        avg_confidence = sum(candidate.confidence for candidate in ordered) / len(ordered)
        return merged_text, round(avg_confidence, 4)

    def extract_text_batch(self, frames: list[object]) -> list[tuple[str, float | None]]:
        if not frames:
            return []

        primary_views = [self._build_primary_view(frame) for frame in frames]
        raw_batches = self._readtext_batched_fast([view.image for view in primary_views])

        results: list[tuple[str, float | None]] = []
        for frame, view, raw_results in zip(frames, primary_views, raw_batches):
            candidates = self._extract_candidates_from_results(raw_results=raw_results, view=view)
            fast_result = self._finalize_candidates(candidates)
            if self._needs_fast_rescue(fast_result):
                secondary_view = self._build_secondary_fast_view(frame)
                candidates.extend(self._extract_from_view_fast(secondary_view))
                fast_result = self._finalize_candidates(candidates)
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

        ordered = sorted(merged, key=lambda item: (item.y_center, item.x_center))
        merged_text = "\n".join(candidate.text for candidate in ordered)
        avg_confidence = sum(candidate.confidence for candidate in ordered) / len(ordered)
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
