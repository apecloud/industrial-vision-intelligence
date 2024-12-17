from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import torch
from cn_clip.clip import load_from_name
import numpy as np
import logging
import sys
from typing import Dict
import os
from datetime import datetime
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class VectorResponse(BaseModel):
    vector: list[float]
    time_taken: float

class ImageVectorizer:
    def __init__(self, model_path: str = None):
        """
        初始化图像向量化器
        
        Args:
            model_path: 模型路径，如果提供则从本地加载模型，否则从预训练库加载
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # 加载模型
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from local path: {model_path}")
                self.model = torch.load(model_path, map_location=self.device)
                # 假设预处理方法与 ViT-B-16 相同，如果不同需要相应调整
                _, self.preprocess = load_from_name("ViT-B-16", device=self.device)
            else:
                logger.info("Loading default ViT-B-16 model")
                self.model, self.preprocess = load_from_name("ViT-B-16", device=self.device)
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def process_image(self, image_data: bytes) -> torch.Tensor:
        """处理图片数据"""
        try:
            # 限制图片大小
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            if image.size[0] * image.size[1] > 25000000:  # 25MP
                raise ValueError("Image too large")
                
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            return image
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def encode_image(self, image: torch.Tensor) -> np.ndarray:
        """编码图片为特征向量"""
        try:
            with torch.no_grad():
                features = self.model.encode_image(image)
                features = features / features.norm(dim=1, keepdim=True)
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise

    async def vectorize(self, image_data: bytes) -> np.ndarray:
        """将图片转换为特征向量"""
        image = self.process_image(image_data)
        vector = self.encode_image(image)
        return vector[0]  # 返回一维数组

class Config:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, model_path: str = None):
        self.host = host
        self.port = port
        self.model_path = model_path

# 创建 FastAPI 应用
app = FastAPI(
    title="Image Vectorization API",
    description="Convert images to feature vectors using ViT-B-16 model",
    version="1.0.0"
)

# 创建向量化器实例和配置实例
vectorizer = None
config = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化向量化器"""
    global vectorizer
    try:
        vectorizer = ImageVectorizer(config.model_path)
    except Exception as e:
        logger.error(f"Failed to initialize vectorizer: {e}")
        sys.exit(1)

@app.post("/vectorize", response_model=VectorResponse)
async def vectorize_image(
    file: UploadFile = File(..., description="Image file to vectorize")
) -> VectorResponse:
    """
    将上传的图片转换为特征向量
    
    Args:
        file: 上传的图片文件
    
    Returns:
        包含特征向量和处理时间的响应
    """
    try:
        start_time = datetime.now()
        
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise ValueError("File must be an image")
        
        # 读取图片数据
        image_data = await file.read()
        
        # 转换为特征向量
        vector = await vectorizer.vectorize(image_data)
        
        # 计算处理时间
        time_taken = (datetime.now() - start_time).total_seconds()
        
        return VectorResponse(
            vector=vector.tolist(),
            time_taken=time_taken
        )
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"Error during vectorization: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """健康检查端点"""
    return {
        "status": "healthy",
        "device": vectorizer.device if vectorizer else "not initialized",
        "model_path": config.model_path if config and config.model_path else "default"
    }

def start_server(host: str = "0.0.0.0", port: int = 8000, model_path: str = None):
    """启动服务器"""
    global config
    config = Config(host=host, port=port, model_path=model_path)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the image vectorization API server')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--model', type=str, help='Path to the local model file')
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        model_path=args.model
    )
