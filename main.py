from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
import cv2
import ollama
import time
from pathlib import Path
import asyncio
import json

app = FastAPI()

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


async def analyze_image(image_path: str, object_str: str):
    """
    圖像分析函數，用於檢測圖像中是否包含指定目標。
    - image_path: 圖像文件的路徑
    - object_str: 目標描述（如“貓”或“車”）
    """
    prompt_str = f"""Please analyze the image and answer the following questions:
    1. Is there a {object_str} in the image?
    2. If yes, describe its appearance and location in the image in detail.
    3. If no, describe what you see in the image instead.
    4. On a scale of 1-10, how confident are you in your answer?

    Please structure your response as follows:
    Answer: [YES/NO]
    Description: [Your detailed description]
    Confidence: [1-10]"""

    try:
        # 使用 Ollama 模型進行圖像分析
        response = await asyncio.to_thread(
            ollama.chat,
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': prompt_str,
                'images': [image_path]
            }]
        )

        response_text = response['message']['content']
        response_lines = response_text.strip().split('\n')

        # 解析分析結果
        answer = None
        description = None
        confidence = 10  # 默認信心分數

        for line in response_lines:
            line = line.strip()
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip().upper()
            elif any(line.lower().startswith(prefix) for prefix in
                     ['description:', 'reasoning:', 'alternative description:']):
                description = line.split(':', 1)[1].strip()
            elif line.lower().startswith('confidence:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 10

        # 返回分析結果
        return answer == "YES" and confidence >= 7, description, confidence
    except Exception as e:
        # 如果出錯，返回默認錯誤信息
        print(f"Error during image analysis: {str(e)}")
        return False, "Error occurred", 0


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
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_video(
        video: UploadFile = File(...),
        object_str: str = Form(...)
):
    """
    分析上傳的視頻，檢測是否包含指定目標。
    - video: 上傳的視頻文件
    - object_str: 用戶描述的目標物體
    """
    try:
        # 保存上傳的視頻到指定目錄
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 為當前任務創建專門的幀圖片存放目錄
        task_frames_dir = FRAMES_DIR / video.filename.split('.')[0]
        task_frames_dir.mkdir(exist_ok=True)

        # 定義生成分析結果的異步函數
        async def generate_results():
            cap = cv2.VideoCapture(str(video_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # 獲取視頻的幀率
            frame_count = 0

            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    if frame_count % fps == 0:  # 每秒處理一幀
                        current_second = frame_count // fps
                        frame_path = os.path.join(task_frames_dir, f"frame_{current_second}.jpg")
                        cv2.imwrite(frame_path, frame)

                        if preprocess_image(frame_path):
                            is_match, description, confidence = await analyze_image(frame_path, object_str)

                            # 返回分析結果
                            result = {
                                "status": "success",
                                "frame": {
                                    "second": current_second,
                                    "is_match": is_match,
                                    "description": description,
                                    "confidence": confidence,
                                    "frame_path": f"/frames/{video.filename.split('.')[0]}/frame_{current_second}.jpg"
                                }
                            }

                            yield json.dumps(result) + "\n"

                    frame_count += 1

            finally:
                cap.release()

        # 返回流式分析結果
        return StreamingResponse(generate_results(), media_type="application/json")

    except Exception as e:
        # 如果出現錯誤，返回錯誤信息
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn

    # 啟動 FastAPI 應用
    uvicorn.run(app, host="140.117.73.146", port=8000)
