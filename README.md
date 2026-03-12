# shorts-visual-transcriber

Instagram Reels / TikTok / YouTube Shorts などのショート動画を対象に、  
画面内テキストをOCRで文字起こしするツールです。  
音声認識ではなく、音声なし・BGMのみ動画の文字情報抽出に特化しています。

## できること

- URL またはローカル動画ファイルを入力
- 必要に応じて `yt-dlp` で動画を取得
- シーン変化や時間間隔にもとづいてキーフレームを抽出
- キーフレームにOCRを実行してテキスト化
- 類似テキストを統合して時系列セグメント化
- `JSON` / `TXT` / `SRT` で出力

## この方式が音声なし動画に向いている理由

音声文字起こしは使わず、映像中の文字を直接抽出するためです。

1. 代表フレームを抽出
2. OCRで画面文字を読み取り
3. タイムスタンプ付きで時系列化

## 必要環境

- Python 3.11 以上
- FFmpeg（`PATH` で実行可能）  
  `yt-dlp` の互換性向上のため推奨
- `yt-dlp`（`requirements.txt` に含む）

依存パッケージのインストール:

```bash
pip install -r requirements.txt
```

## Web UI の使い方

Streamlit を起動:

```bash
streamlit run web_ui.py
```

表示されたURL（通常 `http://localhost:8501`）をブラウザで開き、次を実行します。

- URL/ローカルパスを1行ずつ入力、または動画をアップロード
- サイドバーで抽出・OCR設定を調整
- `文字起こしを実行` を押す
- 結果から `JSON/TXT/SRT` をダウンロード

注: 初回起動時に Streamlit の `Email:` 入力が表示されることがあります。  
空欄のまま Enter で問題ありません。

## CLI の使い方

単体入力:

```bash
python main.py "https://www.tiktok.com/@example/video/1234567890"
```

複数入力:

```bash
python main.py "https://www.instagram.com/reel/XXXXXXXX/" "C:\\videos\\short.mp4"
```

よく使うオプション例:

```bash
python main.py "https://www.youtube.com/shorts/XXXXXXXXXXX" ^
  --output-dir output ^
  --download-dir downloads ^
  --sample-fps 2.5 ^
  --scene-threshold 0.32 ^
  --max-interval-sec 2.5 ^
  --similarity-threshold 0.88 ^
  --langs ja,en
```

ログインが必要な動画向け（cookies）:

```bash
python main.py "https://www.instagram.com/reel/XXXXXXXX/" --cookies-file "C:\\path\\cookies.txt"
```

## 出力ファイル

- `<stem>.json`
  - メタデータとセグメントの構造化データ
- `<stem>.txt`
  - 人が読みやすい時系列テキスト
- `<stem>.srt`
  - 字幕形式のタイムライン

## パラメータ調整の目安

- `--sample-fps` を上げる  
  抽出点が増えて取りこぼし減、処理時間は増加
- `--text-change-threshold` を下げる  
  同一背景内のテキスト切替にも反応しやすくなる
- `--scene-threshold` を下げる  
  シーン変化に敏感になり抽出が増える
- `--max-interval-sec` を下げる  
  変化が少ない動画でも強制抽出が増える
- `--similarity-threshold` を下げる  
  近い文字列をより積極的に統合
- `--min-ocr-confidence` を上げる  
  ノイズは減るが薄い文字を落としやすい

## 注意事項

- プラットフォームの利用規約/APIポリシーにより、URL経由取得に制限がある場合があります。
- 処理対象は権限のあるコンテンツのみ使用してください。
- OCR精度は文字サイズ、コントラスト、動きの強さに影響されます。

## 空出力になったとき

このバージョンでは、セグメントが0件だった場合に自動で高密度再試行します。  
それでも空の場合は、次を調整してください。

- `サンプリングFPS` を上げる（例: `3.0 -> 5.0`）
- `文字変化しきい値` を下げる（例: `0.055 -> 0.03`）
- `OCR最小信頼度` を下げる（例: `0.15 -> 0.08`）
- `OCR言語` を動画内文字に合わせる（例: `ja,en`）

## 誤読補正辞書

プロジェクト直下の `ocr_corrections.json` を自動で読み込みます。  
よく出る誤読を `{"誤読": "正しい文字列"}` の形で追加すると、出力時に補正されます。
