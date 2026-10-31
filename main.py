from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
import cv2
import ollama
from pathlib import Path
import asyncio
import json

app = FastAPI()
OLLAMA_CLIENT = ollama.AsyncClient()
VISION_MODEL = "llama3.2-vision"

# 設置上傳目錄與幀圖片存放目錄
UPLOAD_DIR = Path("uploads")  # 上傳的檔案目錄
FRAMES_DIR = Path("frames")  # 提取幀圖片的目錄

# 如果目錄不存在，則自動創建
UPLOAD_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

# 初始化模板目錄
templates = Jinja2Templates(directory="templates")  # HTML模板文件目錄
# 設置靜態文件路徑
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/frames", StaticFiles(directory="frames"), name="frames")


def log_analysis_event(message: str):
    print(f"[VideoFinder] {message}")


def build_analysis_prompt(object_str: str):
    """Build a prompt that treats the user input as a visual target description."""
    return f"""Please analyze the image for the target below.

Target description: {object_str}

Instructions:
1. Treat the target description as a visual description, even if the wording is informal.
2. Focus on visible clothing color, person attributes, and location in the frame.
3. Answer YES if any person matching the description is visible, even in the background.
4. If the target is not visible, describe what is visible instead.

Please structure your response exactly as:
Answer: [YES/NO]
Description: [Your detailed description]
Confidence: [1-10]"""


def parse_analysis_response(response_text: str):
    """Parse the model response into match fields, tolerating Markdown bold markers."""
    response_text = response_text.replace("**", "")
    response_lines = response_text.strip().split("\n")

    answer = None
    description = None
    confidence = None

    for line in response_lines:
        line = line.strip()
        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip().upper()
        elif any(
            line.lower().startswith(prefix)
            for prefix in ["description:", "reasoning:", "alternative description:"]
        ):
            description = line.split(":", 1)[1].strip()
        elif line.lower().startswith("confidence:"):
            try:
                confidence = int(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = None

    is_match = answer == "YES" and (confidence is None or confidence >= 7)
    return is_match, description, confidence


async def analyze_image(image_path: str, object_str: str):
    """
    圖像分析函數，用於檢測圖像中是否包含指定目標。
    - image_path: 圖像文件的路徑
    - object_str: 目標描述（如“貓”或“車”）
    """
    prompt_str = build_analysis_prompt(object_str)

    try:
        # 使用 Ollama 模型進行圖像分析
        response = await OLLAMA_CLIENT.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt_str,
                'images': [image_path]
            }],
            keep_alive=0,
        )

        response_text = response["message"]["content"]
        return parse_analysis_response(response_text)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # 如果出錯，返回默認錯誤信息
        print(f"Error during image analysis: {str(e)}")
        return False, "Error occurred", 0


async def unload_vision_model():
    """Ask Ollama to release the loaded vision model from memory."""
    try:
        await OLLAMA_CLIENT.generate(model=VISION_MODEL, keep_alive=0)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Error unloading vision model: {str(e)}")


def preprocess_image(image_path):
    """
    圖像預處理函數，用於增強圖像亮度與對比度。
    - image_path: 圖像文件的路徑
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 使用 CLAHE 增強亮度和對比度
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return True


@app.get("/")
async def home(request: Request):
    """
    返回首頁的 HTML 模板。
    """
    return templates.TemplateResponse(request, "index.html")


async def stream_video_analysis(video_path: Path, video_name: str, object_str: str, request: Request | None = None):
    task_frames_dir = FRAMES_DIR / video_name.split('.')[0]
    task_frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
    frame_count = 0
    analyzed_frames = 0
    match_found = False
    match_second = None
    disconnected = False

    try:
        while True:
            if request is not None and await request.is_disconnected():
                disconnected = True
                log_analysis_event("client disconnected, aborting analysis")
                break

            success, frame = cap.read()
            if not success:
                break

            if frame_count % fps == 0:
                current_second = frame_count // fps
                frame_path = os.path.join(task_frames_dir, f"frame_{current_second}.jpg")
                cv2.imwrite(frame_path, frame)

                if preprocess_image(frame_path):
                    is_match, description, confidence = await analyze_image(frame_path, object_str)
                    result = {
                        "status": "success",
                        "frame": {
                            "second": current_second,
                            "is_match": is_match,
                            "description": description,
                            "confidence": confidence,
                            "frame_path": f"/frames/{video_name.split('.')[0]}/frame_{current_second}.jpg"
                        }
                    }

                    yield json.dumps(result) + "\n"
                    analyzed_frames += 1

                    if is_match:
                        match_found = True
                        match_second = current_second
                        log_analysis_event(
                            f"match found at second {current_second}, stopping analysis"
                        )
                        break

            frame_count += 1

        if not disconnected:
            yield json.dumps(
                {
                    "status": "complete",
                    "frames_analyzed": analyzed_frames,
                    "match_found": match_found,
                    "match_second": match_second,
                }
            ) + "\n"
    finally:
        cap.release()
        log_analysis_event(
            "summary: "
            f"frames_analyzed={analyzed_frames}, "
            f"match_found={match_found}, "
            f"match_second={match_second}"
        )
        if not disconnected and not match_found:
            log_analysis_event("analysis complete, unloading model")
        elif match_found:
            log_analysis_event("match handled, unloading model")
        else:
            log_analysis_event("disconnect handled, unloading model")
        await unload_vision_model()


@app.post("/analyze")
async def analyze_video(
        request: Request,
        video: UploadFile = File(...),
        object_str: str = Form(...)
):
    """
    分析上傳的視頻，檢測是否包含指定目標。
    - video: 上傳的視頻文件
    - object_str: 用戶描述的目標物體
    """
    try:
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        return StreamingResponse(
            stream_video_analysis(video_path, video.filename, object_str, request),
            media_type="application/json",
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn

    # 啟動 FastAPI 應用
    uvicorn.run(app, host="127.0.0.1", port=8000)
