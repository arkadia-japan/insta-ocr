from pathlib import Path

from app.runtime_paths import (
    configure_paddle_runtime_env,
    get_runtime_config_dir,
    get_runtime_logs_dir,
    is_ascii_path,
    stage_video_for_runtime,
)


def test_is_ascii_path_detects_non_ascii() -> None:
    assert is_ascii_path(r"C:\temp\video.mp4")
    assert not is_ascii_path(r"D:\バイブコーディング\video.mp4")


def test_stage_video_for_runtime_copies_non_ascii_path(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    source_dir = tmp_path / "日本語"
    source_dir.mkdir()
    source = source_dir / "動画.mp4"
    source.write_bytes(b"video")

    staged = stage_video_for_runtime(source, runtime_dir)

    assert staged != source.resolve()
    assert staged.parent == runtime_dir.resolve()
    assert staged.read_bytes() == b"video"
    assert is_ascii_path(staged.name)


def test_configure_paddle_runtime_env_uses_configured_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "runtime_ascii"
    monkeypatch.setenv("SHORTS_VISUAL_TRANSCRIBER_RUNTIME", str(root))
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    monkeypatch.delenv("PADDLE_OCR_BASE_DIR", raising=False)
    monkeypatch.delenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", raising=False)

    values = configure_paddle_runtime_env()

    assert Path(values["PADDLE_PDX_CACHE_HOME"]).parent == root.resolve()
    assert Path(values["PADDLE_OCR_BASE_DIR"]).parent == root.resolve()
    assert values["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"


def test_runtime_helper_dirs_use_configured_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "runtime_ascii"
    monkeypatch.setenv("SHORTS_VISUAL_TRANSCRIBER_RUNTIME", str(root))

    assert get_runtime_logs_dir() == (root / "logs").resolve()
    assert get_runtime_config_dir() == (root / "config").resolve()
