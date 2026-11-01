# VideoFinder

[English](README.md) | **繁體中文**

[![Tests](https://github.com/yu-ken0207/VideoFinder/actions/workflows/test.yml/badge.svg)](https://github.com/yu-ken0207/VideoFinder/actions/workflows/test.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

VideoFinder 是一個用來從影片中搜尋指定物件的 Python 專案。
它提供一個簡單的 Web 介面，讓使用者上傳影片並輸入想找的目標描述，系統會每秒擷取一幀，交給本機 Ollama 的視覺模型分析，回傳逐幀結果。

## 功能

- 上傳影片並輸入要搜尋的目標描述
- 以 FastAPI 提供 Web 服務
- 使用 OpenCV 每秒擷取一張影格
- 使用 Ollama `llama3.2-vision` 分析圖片內容
- 即時串流回傳逐幀分析結果

## 技術組成

- Python 3.12
- FastAPI
- Uvicorn
- OpenCV
- Ollama Python client
- Jinja2

## 專案結構

```text
VideoFinder/
├── app.py
├── main.py
├── requirements.txt
├── templates/
│   └── index.html
└── uploads/
```

說明：

- `main.py`: 主要 Web 應用入口
- `app.py`: 較早期的本地腳本版本，用來直接分析單支影片
- `templates/index.html`: 前端頁面
- `uploads/`: 上傳影片暫存目錄

## 環境需求

啟動本專案前，請先確認：

1. 已安裝 Conda
2. 已建立名為 `VideoFinder` 的 conda 環境
3. 已安裝本專案 Python 依賴
4. 本機 Ollama 服務已啟動
5. 已下載模型 `llama3.2-vision`

## 建立環境

如果尚未建立環境，可執行：

```bash
conda create -y -n VideoFinder python=3.12 pip
conda activate VideoFinder
pip install -r requirements.txt
```

如果你的 shell 尚未初始化 conda，先執行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate VideoFinder
```

## 安裝 Ollama 模型

請先啟動 Ollama，並拉取視覺模型：

```bash
ollama pull llama3.2-vision
```

若 Ollama 尚未啟動，請先確認本機服務正常運行。

## 啟動專案

進入專案後執行：

```bash
cd VideoFinder
conda activate VideoFinder
uvicorn main:app --host 127.0.0.1 --port 8000
```

啟動後可在瀏覽器開啟：

```text
http://127.0.0.1:8000
```

## 使用方式

1. 開啟首頁
2. 上傳影片檔
3. 輸入想搜尋的目標描述
4. 點擊 `Start Analysis`
5. 系統會顯示每秒擷取的一張影格與分析結果

例如可輸入：

- `a person wearing a red shirt`
- `a man riding a bicycle`
- `a white car`

## API

### `GET /`

回傳首頁 HTML。

### `POST /analyze`

上傳影片並開始分析。

表單欄位：

- `video`: 影片檔案
- `object_str`: 要搜尋的目標描述

回應方式：

- 使用串流 JSON 回傳逐幀分析結果

## 分析流程

1. 接收上傳影片
2. 將影片存到 `uploads/`
3. 使用 OpenCV 每秒取一幀
4. 對影格做基礎影像前處理
5. 呼叫 Ollama 模型分析該影格
6. 將分析結果以串流方式回傳到前端

模型回答 YES 且信心 9 分以上（模型沒給分數也算）才算找到；找到第一張就停止，分析完會把模型從記憶體卸載。

## 注意事項

- 分析速度取決於影片長度、硬體效能與 Ollama 模型速度
- 本專案目前每秒只分析一張影格
- 若 `llama3.2-vision` 未安裝或 Ollama 未啟動，分析會失敗
- `main.py` 內使用本機資料夾 `uploads/` 與 `frames/` 儲存中間檔

## 常見問題

### 1. `conda activate VideoFinder` 無法使用

先執行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate VideoFinder
```

### 2. 啟動後分析失敗

請檢查：

- Ollama 是否正在執行
- `llama3.2-vision` 是否已下載
- 上傳影片格式是否可被 OpenCV 正常讀取

### 3. Python import 行為異常

如果你的 shell 有額外注入 `PYTHONPATH`，可能會混入其他系統套件。必要時可先執行：

```bash
unset PYTHONPATH
```

再重新啟動環境。

## 測試

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
```

測試把 Ollama 換成假的回答，不用真的模型：檢查回答的解析、命中門檻、逐張串流的結果。每次 push 會在 GitHub Actions 自動跑。

## 開發備註

- 主要入口為 `main.py`
- 若要直接執行腳本版流程，可參考 `app.py`

## License

Apache-2.0，見 [LICENSE](LICENSE)。
