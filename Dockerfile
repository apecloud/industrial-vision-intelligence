FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libfontconfig1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    opencv-python \
    fastapi \
    uvicorn[standard] \
    numpy \
    ultralytics \
    python-multipart \
    pydantic \
    pydantic-settings

# 创建工作目录
WORKDIR /app

# 复制模型处理代码
COPY ./web_yolo.py /app/

# 创建必要目录
RUN mkdir -p /mnt/models

ENV YOLO_MODEL_PATH=/mnt/models/best.pt
ENV PYTHONUNBUFFERED=1
ENV WORKERS=4
ENV TIMEOUT=300
ENV PORT=8000
ENV HOST=0.0.0.0

# 创建启动脚本
RUN echo '#!/bin/bash\n\
uvicorn web_yolo:app \
--host ${HOST} \
--port ${PORT} \
--workers ${WORKERS} \
--timeout-keep-alive ${TIMEOUT}' > /app/start.sh \
&& chmod +x /app/start.sh

# 创建非 root 用户
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 设置启动命令
CMD ["/app/start.sh"]
