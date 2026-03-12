from __future__ import annotations

from dataclasses import dataclass


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

    def to_dict(self) -> dict:
        return {
            "input_ref": self.input_ref,
            "platform": self.platform,
            "video_path": self.video_path,
            "duration_sec": round(self.duration_sec, 3),
            "sampled_frames": self.sampled_frames,
            "ocr_hits": self.ocr_hits,
            "fallback_used": self.fallback_used,
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
