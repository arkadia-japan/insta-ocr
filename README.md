# shorts-visual-transcriber

Instagram Reel / TikTok / YouTube Shorts のような縦動画から全文を作成するツールです。

- `Instagram Reel全文` タブは音声認識 (`faster-whisper`) を使います
- `汎用OCR` タブは画面内テキスト抽出 (`PaddleOCR`) を使います
- Web UI は `Streamlit`、動画取得は `yt-dlp`、推奨 Python は `3.11` です

## 実行前提

このプロジェクトは日本語パスによるネイティブ依存の不具合を避けるため、実行時ファイルを ASCII パスへ逃がします。

既定の runtime ディレクトリ:

- `C:\Users\Public\shorts_visual_transcriber_runtime\downloads`
- `C:\Users\Public\shorts_visual_transcriber_runtime\output`
- `C:\Users\Public\shorts_visual_transcriber_runtime\logs`
- `C:\Users\Public\shorts_visual_transcriber_runtime\paddlex_cache`
- `C:\Users\Public\shorts_visual_transcriber_runtime\paddleocr`
- `C:\Users\Public\shorts_visual_transcriber_runtime\faster_whisper`

## クイックスタート

最初は `start_web_ui.bat` を使うのが最も安全です。

```powershell
cd "D:\バイブコーディング\shorts-visual-transcriber"
cmd /c start_web_ui.bat
```

`start_web_ui.bat` は次を行います。

1. `Python 3.11` と `.venv311` を確認
2. 必要依存を確認し、足りなければインストール
3. Paddle 系キャッシュを `C:\Users\Public\shorts_visual_transcriber_runtime` に向ける
4. `localhost:8501` の既存 Web UI を検知
5. `/_stcore/health` が `ok` になってからブラウザを開く
6. ログを `C:\Users\Public\shorts_visual_transcriber_runtime\logs\web_ui_latest.log` に出力

## 手動セットアップ

```powershell
cd "D:\バイブコーディング\shorts-visual-transcriber"
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m streamlit run web_ui.py
```

## Web UI の使い方

1. ブラウザで `http://localhost:8501` を開く
2. `Instagram Reel全文` か `汎用OCR` を選ぶ
3. 入力を貼る
4. 実行する

`Instagram Reel全文`

- `https://www.instagram.com/reel/.../` の URL を入力します
- 音声トラックを `faster-whisper` で文字起こしします
- 出力は `JSON` / `TXT` / `SRT` です

`汎用OCR`

- URL またはローカル動画を入力します
- 画面内テキストを OCR で抽出します
- 出力は `JSON` / `TXT` / `SRT` です

## CLI

```powershell
cd "D:\バイブコーディング\shorts-visual-transcriber"
.\.venv311\Scripts\python.exe main.py "https://www.instagram.com/reel/XXXXXXXX/"
```

## Google Sheets 連携

`投稿データ260301` のようなシートをキューとして使う場合は、ローカルの Python から Sheets API を叩く形が一番安定します。

前提:

- Google Cloud で service account を作成する
- 対象スプレッドシートをその service account に共有する
- 動画URL列の列記号を把握する

実行例:

```powershell
cd "D:\バイブコーディング\shorts-visual-transcriber"
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m app.sheets_sync `
  --service-account-file "C:\secure\service-account.json" `
  --spreadsheet-id "1kFFRfOkcgtp0a4SZZ5q5hxMXLNAwlAGO6-bCEROAgCo" `
  --sheet-name "投稿データ260301" `
  --url-column "J"
```

このコマンドは次を行います。

- K列 `スタイル` が `音声リール` の行は ASR を実行する
- K列 `スタイル` が `画像リール` の行は OCR を実行する
- L列 `文字起こし` に全文を書き戻す
- M列に状態、N列に処理日時、O列にエラーを記録する

自動運転する場合は、このコマンドを Windows Task Scheduler で数分おきに実行してください。

## トラブルシュート

- Web UI が起動しない場合は `C:\Users\Public\shorts_visual_transcriber_runtime\logs\web_ui_latest.log` を確認してください
- 音声トラックのない Reel では ASR は失敗します
- 一部の Instagram URL は未ログインだと `yt-dlp` が取得できません。その場合は cookies 指定が必要です
- Windows では `paddle` を先に import してから `ctranslate2` / `faster-whisper` を import すると `WinError 127` になる環境があります
- このリポジトリではその回避のため、ASR 側で `faster-whisper` を先に preload しています
