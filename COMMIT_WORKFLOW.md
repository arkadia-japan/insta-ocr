# Commit Workflow

このリポジトリでは `prepare-commit-msg` フックでコミットメッセージを自動整形します。

形式:

```text
YYYY-MM-DD 自動要約
YYYY-MM-DD 自動要約 | 手入力メモ
```

例:

```text
2026-03-12 OCR・セグメント判定更新
2026-03-12 OCR・セグメント判定更新 | しきい値を調整
```

使い方:

```powershell
git add .
git commit -m "しきい値を調整"
```

この場合、実際のコミット件名は自動で次のようになります。

```text
2026-03-12 OCR・セグメント判定更新 | しきい値を調整
```

手入力を省略したい場合:

```powershell
git add .
git commit
```

エディタが開いた時点で、1行目に自動件名が入っています。

要約ルール:
- `app/ocr_engine.py`, `app/text_corrections.py`, `ocr_corrections.json`: `OCR`
- `app/frame_sampler.py`: `セグメント判定`
- `app/pipeline.py` など: `パイプライン`
- `web_ui.py`: `Web UI`
- `app/cli.py`, `main.py`: `CLI`
- `tests/`: `テスト`
- `.githooks/`, `app/commit_messages.py`: `Git運用`
- `pyproject.toml`, `requirements.txt`, `.gitignore`: `設定`
- `README.md`: `ドキュメント`
