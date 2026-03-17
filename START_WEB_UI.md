# Web UI の起動

## 推奨手順

1. `D:\バイブコーディング\shorts-visual-transcriber` を開く
2. `start_web_ui.bat` をダブルクリックする
3. ブラウザで `http://localhost:8501` を開く

`Sheetsモード` `Whisperモード` `OCRモード` を使い分けます。

- `Sheetsモード`
  Google Sheets の対象タブを選び、未処理実行や画像リール再実行をボタンで操作できます
- `Whisperモード`
  音声認識で Instagram Reel の全文を作成します
- `OCRモード`
  画面テキスト抽出を行います

## この起動スクリプトが行うこと

- `.venv311` を優先して使う
- `streamlit -> faster_whisper -> cv2 -> yt_dlp -> numpy -> paddleocr -> paddle` の順で依存チェックする
- `localhost:8501` の既存 Web UI を検知する
- `C:\Users\Public\shorts_visual_transcriber_runtime` を runtime ルートに使う
- `/_stcore/health` が `ok` になるまで待ってからブラウザを開く
- ログを `C:\Users\Public\shorts_visual_transcriber_runtime\logs\web_ui_latest.log` に出す
- 起動に失敗してもウィンドウを閉じず、そのまま原因を確認できるようにする

## うまく起動しない場合

- 古い Streamlit / Python プロセスが残っていないか確認する
- `C:\Users\Public\shorts_visual_transcriber_runtime\logs\web_ui_latest.log` を確認する
- すでに `8501` が使われている場合は既存 Web UI の再利用が優先される
- `Whisperモード` でエラーが出る場合は、URL が `https://www.instagram.com/reel/.../` 形式か確認する
- 音声トラックのない Reel は文字起こしできない

## 停止方法

- 起動した `cmd` ウィンドウで `Ctrl + C`
