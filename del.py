import cv2
import numpy as np

video_path = 'test_video.mp4'
width, height = 640, 480
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'X264')

# 创建一个 VideoWriter 对象
writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

if writer.isOpened():
    print("X264 encoder is supported!")
    # 创建一帧白色图像
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    for _ in range(30):  # 写入 30 帧
        writer.write(frame)
    writer.release()  # 释放资源
else:
    print("X264 encoder is NOT supported.")
