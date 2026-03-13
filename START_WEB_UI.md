# Web UI の簡単な起動方法

## いちばん簡単な方法

プロジェクト直下にある `start_web_ui.bat` をダブルクリックしてください。

初回:
- Python の確認
- 仮想環境 `.venv` の作成
- 必要パッケージのインストール

2回目以降:
- `start_web_ui.bat` をダブルクリックするだけで起動できます

## 起動後

ブラウザで次のURLが開きます。

```text
http://localhost:8501
```

もし自動で開かない場合は、手動で上のURLを開いてください。

## 終了方法

起動した黒い画面で `Ctrl + C` を押してください。

## PowerShell から起動する場合

```powershell
cd "D:\バイブコーディング\shorts-visual-transcriber"
.\start_web_ui.bat
```

## 補足

- 依存パッケージが足りない場合は、起動スクリプトが自動で `requirements.txt` を使って補います
- 仮想環境は `.venv` に作られます
