from __future__ import annotations

from dataclasses import dataclass

from .utils import format_compact_time


@dataclass
class Snapshot:
    time_sec: float
    text: str
    confidence: float | None = None


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    confidence: float | None = None


@dataclass
class TranscriptionResult:
    input_ref: str
    platform: str
    video_path: str
    duration_sec: float
    sampled_frames: int
    ocr_hits: int
    fallback_used: bool
    segments: list[TranscriptSegment]
    transcription_mode: str = "ocr"

    def to_full_text(self, include_timestamps: bool = False) -> str:
        blocks: list[str] = []
        previous_text = ""
        for segment in self.segments:
            text = segment.text.strip()
            if not text or text == previous_text:
                continue
            if include_timestamps:
                blocks.append(
                    f"[{format_compact_time(segment.start_sec)} - {format_compact_time(segment.end_sec)}]\n{text}"
                )
            else:
                blocks.append(text)
            previous_text = text
        return "\n\n".join(blocks)

    def to_dict(self) -> dict:
        return {
            "input_ref": self.input_ref,
            "platform": self.platform,
            "video_path": self.video_path,
            "duration_sec": round(self.duration_sec, 3),
            "sampled_frames": self.sampled_frames,
            "ocr_hits": self.ocr_hits,
            "fallback_used": self.fallback_used,
            "transcription_mode": self.transcription_mode,
            "segments": [
                {
                    "start_sec": round(segment.start_sec, 3),
                    "end_sec": round(segment.end_sec, 3),
                    "text": segment.text,
                    "confidence": segment.confidence,
                }
                for segment in self.segments
            ],
        }
