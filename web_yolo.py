from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Dict
import io
from PIL import Image
import argparse
import os
from pydantic_settings import BaseSettings

# 配置类
class Settings(BaseSettings):
    MODEL_PATH: str = "/root/best.pt"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    WORKERS: int = 1
    CONFIDENCE_THRESHOLD: float = 0.5
    DEVICE: str = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    class Config:
        env_prefix = "YOLO_"  # 环境变量前缀

# 创建配置实例
settings = Settings()

app = FastAPI(
    title="Industrial Defect Detection API",
    description="YOLOv8 model service for defect detection"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DefectDetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(settings.DEVICE)
        print(f"Model loaded from {model_path} using {settings.DEVICE}")
        
    async def predict(self, image: np.ndarray) -> List[Dict]:
        results = self.model(
            image, 
            conf=settings.CONFIDENCE_THRESHOLD,
            device=settings.DEVICE
        )[0]
        
        detections = []
        for box in results.boxes:
            detection = {
                "class": int(box.cls),
                "class_name": results.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]]
            }
            detections.append(detection)
            
        return detections

# 全局模型实例
detector = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化模型"""
    global detector
    detector = DefectDetector(settings.MODEL_PATH)

@app.post("/detect")
async def detect_defects(file: UploadFile = File(...)):
    """
    接收图片文件并进行缺陷检测
    返回检测到的缺陷列表，包含类别、置信度和边界框坐标
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        detections = await detector.predict(image)
        
        return {
            "status": "success",
            "message": f"Detected {len(detections)} defects",
            "detections": detections,
            "model_path": settings.MODEL_PATH
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/health")
async def health_check():
    """服务健康检查接口"""
    return {
        "status": "healthy",
        "model_path": settings.MODEL_PATH,
        "device": settings.DEVICE
    }

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Service")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model file")
    parser.add_argument("--port", type=int, default=8000, help="Service port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service host")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 更新配置
    settings.MODEL_PATH = args.model
    settings.PORT = args.port
    settings.HOST = args.host
    settings.WORKERS = args.workers
    
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=True
    )
