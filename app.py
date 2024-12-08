import cv2
import os
import ollama
import time


def analyze_image(image_path, object_str):
    """
    分析單張圖像，檢測是否存在目標對象
    Args:
        image_path: 圖像文件路徑
        object_str: 要檢測的目標對象描述

    Returns:
        tuple: (是否匹配, 描述文本, 置信度)
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
        # 調用llama模型分析圖像
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[
                {
                    'role': 'user',
                    'content': prompt_str,
                    'images': [image_path]
                }
            ]
        )

        print(f"等待模型分析中...")
        time.sleep(1)  # 短暫延遲確保響應完整

        # 獲取並打印原始響應
        response_text = response['message']['content']
        print(f"Raw response: {response_text}")

        # 處理響應文本，移除Markdown格式符號
        response_text = response_text.replace('**', '')
        response_lines = response_text.strip().split('\n')

        # 從響應中提取關鍵信息
        answer = None
        description = None
        confidence = 10  # 預設置信度為10，因為模型沒有明確返回置信度

        # 逐行解析響應內容
        for line in response_lines:
            line = line.strip()
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip().upper()
            # 同時匹配Description、Reasoning和Alternative Description
            elif any(line.lower().startswith(prefix) for prefix in
                     ['description:', 'reasoning:', 'alternative description:']):
                description = line.split(':', 1)[1].strip()
            elif line.lower().startswith('confidence:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip())
                except ValueError:
                    confidence = 10  # 如果無法解析置信度，使用預設值

        # 檢查是否獲取到必要的信息
        if answer is None or description is None:
            raise ValueError("Response format is incomplete")

        print(f"解析結果 - 答案: {answer}, 描述: {description}, 置信度: {confidence}")

        # 返回分析結果
        return answer == "YES" and confidence >= 7, description, confidence
    except Exception as e:
        print(f"圖像分析時發生錯誤: {e}")
        import traceback
        print(traceback.format_exc())
        return False, "Error occurred", 0


def preprocess_image(image_path):
    """
    圖像預處理函數，增強圖像質量
    Args:
        image_path: 圖像文件路徑
    """
    # 讀取圖像
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤: 無法讀取圖像文件 {image_path}")
        return

    # 轉換顏色空間並進行對比度增強
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 保存處理後的圖像
    cv2.imwrite(image_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])


def extract_and_analyze_frames(video_path, output_folder, object_str):
    """
    從視頻中提取幀並分析是否包含目標對象
    Args:
        video_path: 視頻文件路徑
        output_folder: 幀圖像保存文件夾
        object_str: 要檢測的目標對象描述

    Returns:
        int or None: 找到目標的時間點（秒），未找到返回None
    """
    # 創建輸出目錄
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打開視頻文件
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"錯誤: 無法打開視頻文件 {video_path}")
        return None

    # 獲取視頻FPS
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    consecutive_matches = 0
    match_threshold = 1  # 連續匹配閾值
    cool_down_time = 2  # 每幀分析後的冷卻時間（秒）

    print(f"開始分析視頻，FPS: {fps}")

    try:
        while True:
            # 讀取視頻幀
            success, frame = video.read()
            if not success:
                break

            # 每秒處理一幀
            if frame_count % fps == 0:
                print(f"\n處理第 {frame_count // fps} 秒的幀")

                # 保存當前幀
                output_filename = os.path.join(output_folder, f"frame_{frame_count // fps}.jpg")
                output_filename = os.path.abspath(output_filename)

                cv2.imwrite(output_filename, frame)
                print(f"已保存幀到: {output_filename}")

                # 預處理圖像
                preprocess_image(output_filename)
                print("已完成圖像預處理")

                print("開始分析圖像...")
                print(f"使用圖像路徑: {output_filename}")

                # 檢查文件是否存在
                if not os.path.exists(output_filename):
                    print(f"警告: 文件不存在: {output_filename}")
                    continue

                # 分析圖像
                is_match, description, confidence = analyze_image(output_filename, object_str)
                print(f"分析完成 - 匹配: {is_match}, 置信度: {confidence}")
                print(f"描述: {description}")

                # 處理匹配結果
                if is_match:
                    consecutive_matches += 1
                    print(f"潛在匹配 - 時間: 第 {frame_count // fps} 秒")
                    print(f"描述: {description}")
                    print(f"置信度: {confidence}")

                    # 如果連續匹配次數達到閾值，返回結果並退出
                    if consecutive_matches >= match_threshold:
                        match_time = frame_count // fps - match_threshold + 1
                        print(f"找到連續匹配！時間: 第 {match_time} 秒到第 {frame_count // fps} 秒")
                        video.release()  # 釋放視頻資源
                        return match_time  # 直接返回結果
                else:
                    consecutive_matches = 0

                # 分析完一幀後的冷卻時間
                print(f"等待 {cool_down_time} 秒進行顯卡冷卻...")
                time.sleep(cool_down_time)

            frame_count += 1

    finally:
        # 確保視頻資源被釋放
        video.release()

    print(f"未找到匹配的圖像。共分析了 {frame_count // fps} 張圖像。")
    return None


# 主程序入口
if __name__ == "__main__":
    # 設置參數
    video_path = "./a.mp4"
    output_folder = "output_frames"
    object_to_find = "A man riding a bicycle"

    print("開始運行視頻分析程序...")
    # 運行分析
    result = extract_and_analyze_frames(video_path, output_folder, object_to_find)

    # 輸出結果
    if result is not None:
        print(f"目標對象在視頻的第 {result} 秒被找到。")
    else:
        print("在整個視頻中未找到目標對象。")
