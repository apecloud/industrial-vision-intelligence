from ultralytics import YOLO
import torch

# For CPU:
device = 'cpu'

# Or for Apple Silicon (M1/M2/M3) Macs:
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 考虑到缺陷的精细程度，建议使用medium或large版本
model = YOLO('yolov8s.pt')  # 或 yolov8l.pt

# 训练配置
results = model.train(
    data='./dataset/data.yaml',
    epochs=20,
    imgsz=640,  # 更大的图像尺寸以捕捉细节
    batch=8,
    patience=50,  # 早停
    save=True,
    device=device,
    # 学习率设置
    lr0=0.01,
    lrf=0.001,
    # 损失权重
    box=0.05,
    cls=0.3,
    dfl=1.0,
)
