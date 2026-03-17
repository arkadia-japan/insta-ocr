from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import TranscriptSegment
from .runtime_paths import get_runtime_root
from .utils import normalize_text

try:  # Preload ctranslate2 before Paddle to avoid OpenMP DLL conflicts on Windows.
    from faster_whisper import WhisperModel as FasterWhisperModel
    _FASTER_WHISPER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local native runtime
    FasterWhisperModel = None
    _FASTER_WHISPER_IMPORT_ERROR = exc


@dataclass
class AudioTranscriptionInfo:
    language: str | None
    language_probability: float | None
    duration_sec: float


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        cpu_threads: int = 4,
    ) -> None:
        if FasterWhisperModel is None:
            raise RuntimeError(
                "Audio transcription backend could not be initialized. "
                f"faster-whisper import failed: {_FASTER_WHISPER_IMPORT_ERROR!r}"
            ) from _FASTER_WHISPER_IMPORT_ERROR

        model_root = get_runtime_root() / "faster_whisper"
        model_root.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.language = language
        self.model = FasterWhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            download_root=str(model_root),
        )

    def transcribe_video(self, video_path: Path) -> tuple[list[TranscriptSegment], AudioTranscriptionInfo]:
        if _has_audio_stream(video_path) is False:
            raise RuntimeError("Audio transcription failed: no audio stream was found in the input video.")

        try:
            raw_segments, raw_info = self.model.transcribe(
                str(video_path),
                task="transcribe",
                language=self.language,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                vad_filter=True,
            )
        except Exception as exc:  # pragma: no cover - backend depends on environment
            raise RuntimeError(f"Audio transcription failed: {exc}") from exc

        segments: list[TranscriptSegment] = []
        for raw_segment in raw_segments:
            text = normalize_text(getattr(raw_segment, "text", ""))
            if not text:
                continue
            start_sec = max(0.0, float(getattr(raw_segment, "start", 0.0)))
            end_sec = max(start_sec + 0.1, float(getattr(raw_segment, "end", start_sec + 0.1)))
            segments.append(
                TranscriptSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=text,
                    confidence=None,
                )
            )

        info = AudioTranscriptionInfo(
            language=getattr(raw_info, "language", None),
            language_probability=getattr(raw_info, "language_probability", None),
            duration_sec=float(getattr(raw_info, "duration", 0.0) or 0.0),
        )
        return segments, info


def _has_audio_stream(video_path: Path) -> bool | None:
    try:
        import av
    except Exception:
        return None

    try:
        with av.open(str(video_path)) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return None
