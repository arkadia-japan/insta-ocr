from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+")
SPECIAL_SUBJECT_PREFIXES = ("fixup!", "squash!", "merge ")


@dataclass(frozen=True)
class CategoryRule:
    label: str
    prefixes: tuple[str, ...]


CATEGORY_RULES = (
    CategoryRule("OCR", ("app/ocr_engine.py", "app/text_corrections.py", "ocr_corrections.json")),
    CategoryRule("セグメント判定", ("app/frame_sampler.py",)),
    CategoryRule("パイプライン", ("app/pipeline.py", "app/models.py", "app/exporters.py", "app/downloader.py")),
    CategoryRule("Web UI", ("web_ui.py",)),
    CategoryRule("CLI", ("app/cli.py", "main.py")),
    CategoryRule("テスト", ("tests/",)),
    CategoryRule("Git運用", (".githooks/", "app/commit_messages.py")),
    CategoryRule("設定", ("pyproject.toml", "requirements.txt", ".gitignore")),
    CategoryRule("ドキュメント", ("README.md",)),
)


def build_auto_summary(paths: list[str]) -> str:
    normalized_paths = [path.replace("\\", "/") for path in paths if path.strip()]
    if not normalized_paths:
        return "メンテナンス更新"

    counts: Counter[str] = Counter()
    unmatched_count = 0
    for path in normalized_paths:
        matched = False
        for rule in CATEGORY_RULES:
            if any(path == prefix or path.startswith(prefix) for prefix in rule.prefixes):
                counts[rule.label] += 1
                matched = True
        if not matched:
            unmatched_count += 1

    ordered_labels = [
        rule.label
        for rule in CATEGORY_RULES
        if counts[rule.label] > 0
    ]

    if unmatched_count:
        ordered_labels.append("その他")

    labels = ordered_labels[:3]
    if len(ordered_labels) > 3:
        labels.append("他")

    return "・".join(labels) + "更新"


def format_commit_subject(
    existing_subject: str,
    auto_summary: str,
    commit_date: str | None = None,
) -> str:
    subject = existing_subject.strip()
    dated = commit_date or date.today().isoformat()
    lowered = subject.lower()

    if subject and DATE_PREFIX_RE.match(subject):
        return subject
    if subject and lowered.startswith(SPECIAL_SUBJECT_PREFIXES):
        return subject
    if not subject:
        return f"{dated} {auto_summary}"
    if subject == auto_summary or auto_summary in subject:
        return f"{dated} {subject}"
    return f"{dated} {auto_summary} | {subject}"


def rewrite_commit_message_text(
    original_text: str,
    auto_summary: str,
    commit_date: str | None = None,
) -> str:
    lines = original_text.splitlines()
    subject_index = _first_subject_index(lines)
    existing_subject = lines[subject_index] if subject_index is not None else ""
    new_subject = format_commit_subject(existing_subject, auto_summary, commit_date=commit_date)

    if subject_index is None:
        updated_lines = [new_subject, ""]
        updated_lines.extend(lines)
        return "\n".join(updated_lines).rstrip() + "\n"

    lines[subject_index] = new_subject
    return "\n".join(lines).rstrip() + "\n"


def get_staged_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def apply_prepare_commit_msg_hook(message_file: Path, source: str | None = None) -> None:
    if source in {"merge"}:
        return

    original_text = message_file.read_text(encoding="utf-8")
    existing_subject = _extract_subject(original_text)
    if existing_subject.lower().startswith(SPECIAL_SUBJECT_PREFIXES):
        return

    repo_root = Path.cwd()
    auto_summary = build_auto_summary(get_staged_paths(repo_root))
    updated_text = rewrite_commit_message_text(original_text, auto_summary=auto_summary)
    message_file.write_text(updated_text, encoding="utf-8")


def _extract_subject(text: str) -> str:
    lines = text.splitlines()
    subject_index = _first_subject_index(lines)
    if subject_index is None:
        return ""
    return lines[subject_index].strip()


def _first_subject_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return index
    return None


def _main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] != "hook":
        return 0

    message_file = Path(argv[2]).resolve()
    source = argv[3] if len(argv) >= 4 else None
    apply_prepare_commit_msg_hook(message_file=message_file, source=source)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
