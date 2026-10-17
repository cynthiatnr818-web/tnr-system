import cv2
import time
import os
import sys

RTSP_URL = "rtsp://username:password@ip_address:port/stream"  # 替换为实际的 RTSP 地址
OUTPUT_FOLDER = "./captured_images"  # 截图保存路径
INTERVAL = 5  # 截图间隔（秒）
MODEL_PATH = "model_checkpoint.pth"  # 模型文件路径

# 创建输出文件夹
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def capture_and_process_video():
    """从视频流中截图并调用猫脸识别模型"""
    # 加载模型
    model = CatRecognitionModel(num_classes=2)  # 假设是二分类问题：是否为目标猫
    checkpoint = load_model_parameters(model, MODEL_PATH)
    model.eval()  # 将模型设置为评估模式
    
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("无法打开视频流，请检查 RTSP URL")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * INTERVAL)
    frame_count = 0
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，检查连接状态")
            break

        if frame_count % frame_interval == 0:
            timestamp = int(time.time())
            image_path = os.path.join(OUTPUT_FOLDER, f"frame_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

            if last_frame is not None:
                # 计算帧差异
                frame_diff = cv2.absdiff(last_frame, frame)
                gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                activity_detected = cv2.countNonZero(thresh) > 1000

                if activity_detected:
                    print("检测到活动，进行猫脸识别")
                    try:
                        # 使用加载的模型进行预测
                        is_similar = predict_image(
                            model,
                            image_path,
                            checkpoint['train_features'],
                            checkpoint['train_labels']
                        )

                        # 根据预测结果进行处理
                        if is_similar:
                            print("输入图像与数据库中有足够相似的图片。")  # 这里可以添加后续处理逻辑

                    except Exception as e:
                        print(f"调用猫脸识别模型时出错：{e}")

            last_frame = frame

        frame_count += 1

    cap.release()

if __name__ == "__main__":
    capture_and_process_video()
