# VideoFinder

**English** | [繁體中文](README.zh-TW.md)

[![Tests](https://github.com/yu-ken0207/VideoFinder/actions/workflows/test.yml/badge.svg)](https://github.com/yu-ken0207/VideoFinder/actions/workflows/test.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Upload a video and describe what you are looking for (for example "a person wearing a red shirt"). VideoFinder samples one frame per second, asks a local vision model through Ollama (llama3.2-vision) whether the target appears, and streams the per-frame results to the web page. Built with FastAPI and OpenCV; the video never leaves your machine.

## Features

- Upload a video and describe the target to find
- Web service built with FastAPI
- One frame per second extracted with OpenCV
- Each frame analyzed by Ollama `llama3.2-vision`
- Per-frame results streamed back as they come in

## Tech stack

- Python 3.12
- FastAPI
- Uvicorn
- OpenCV
- Ollama Python client
- Jinja2

## Project structure

```text
VideoFinder/
├── app.py
├── main.py
├── requirements.txt
├── templates/
│   └── index.html
└── uploads/
```

- `main.py`: the web app entry point
- `app.py`: an earlier local script that analyzes a single video directly
- `templates/index.html`: the web page
- `uploads/`: temporary folder for uploaded videos

## Requirements

Before starting, make sure that:

1. Conda is installed
2. A conda environment named `VideoFinder` exists
3. The Python dependencies are installed
4. Ollama is running on this machine
5. The `llama3.2-vision` model is downloaded

## Create the environment

```bash
conda create -y -n VideoFinder python=3.12 pip
conda activate VideoFinder
pip install -r requirements.txt
```

If your shell hasn't initialized conda yet:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate VideoFinder
```

## Download the Ollama model

Start Ollama and pull the vision model:

```bash
ollama pull llama3.2-vision
```

## Run

```bash
cd VideoFinder
conda activate VideoFinder
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open in your browser:

```text
http://127.0.0.1:8000
```

## How to use

1. Open the home page
2. Upload a video file
3. Describe the target you want to find
4. Click `Start Analysis`
5. The page shows each extracted frame (one per second) with its result

For example:

- `a person wearing a red shirt`
- `a man riding a bicycle`
- `a white car`

## API

### `GET /`

Returns the home page HTML.

### `POST /analyze`

Uploads a video and starts the analysis.

Form fields:

- `video`: the video file
- `object_str`: description of the target

Response:

- Per-frame results as streamed JSON

## How it works

1. Receive the uploaded video
2. Save it to `uploads/`
3. Extract one frame per second with OpenCV
4. Do basic image preprocessing on the frame
5. Ask the Ollama model about the frame
6. Stream the result back to the page

A frame counts as found only when the model answers YES with confidence 9 or higher (or gives no confidence score). The analysis stops at the first match, and the model is unloaded from memory afterwards.

## Notes

- Speed depends on the video length, your hardware, and the Ollama model
- Only one frame per second is analyzed
- The analysis fails if `llama3.2-vision` isn't installed or Ollama isn't running
- `main.py` stores intermediate files in the local `uploads/` and `frames/` folders

## Troubleshooting

### 1. `conda activate VideoFinder` doesn't work

Run first:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate VideoFinder
```

### 2. Analysis fails after starting

Check:

- Is Ollama running?
- Is `llama3.2-vision` downloaded?
- Can OpenCV read the video format?

### 3. Strange Python import errors

If your shell sets an extra `PYTHONPATH`, packages from elsewhere may get mixed in. If needed:

```bash
unset PYTHONPATH
```

Then activate the environment again.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
```

The tests replace Ollama with fake answers, so no real model is needed. They check answer parsing, the match threshold, and the streamed per-frame results. GitHub Actions runs them on every push.

## Development notes

- The main entry point is `main.py`
- For the script version of the flow, see `app.py`

## License

Apache-2.0, see [LICENSE](LICENSE).
