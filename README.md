# industrial-vision-intelligence

## Introduction
IVI System is an innovative solution that combines traditional deep learning models, vision-language models, and large language models to address quality control and knowledge management challenges in industrial manufacturing. This project leverages the strengths of YOLOv8, CogVLM, Qwen-72B, and ViT-B-16 to create a comprehensive system for defect detection, analysis, and knowledge extraction.

We demonstrate the system's effectiveness using the publicly available Magnetic Tile Defect Dataset(MTDD) https://github.com/abin24/Magnetic-tile-defect-datasets. as a benchmark case study. The implementation workflow consists of three main components:

### Defect Detection Pipeline:
  
- Trained YOLOv8 model on MTDD for defect localization
Model outputs include defect classifications and bounding box coordinates
Achieves real-time detection capabilities for various defect categories

### Knowledge Retrieval System:
  
- Historical defect data and associated knowledge stored in a vector database
Implements similarity-based image retrieval using ViT-B-16 embeddings
Enables efficient querying of relevant historical cases and expertise

### Intelligent Analysis Integration:
  
- Combines current detection results with retrieved historical data
Utilizes carefully crafted prompt templates for context structuring
Leverages LLM capabilities to generate comprehensive analysis reports
This integrated approach enables automated defect analysis while incorporating historical knowledge, resulting in human-readable summaries that facilitate decision-making in industrial quality control processes. The system demonstrates the practical application of combining traditional computer vision techniques with modern AI capabilities for industrial inspection tasks.

## Architecture
![image](https://github.com/user-attachments/assets/7c59e18f-2e49-45f6-9d99-5a297383f799)


## Key Features
### 1. Multi-Model Integration
YOLOv8 for real-time object detection and defect identification
CogVLM for detailed visual understanding and reasoning
Qwen-72B for natural language processing and knowledge extraction
ViT-B-16 for image embedding and similarity search
### 2. Quality Control Pipeline
Automated defect detection and classification
Visual anomaly analysis
Historical pattern recognition
Real-time quality monitoring
### 3. Knowledge Management
Experience capture and digitalization
Visual-textual knowledge base construction
Intelligent defect analysis reporting
Solution recommendation system

## Use Cases
### Manufacturing Quality Control
- Real-time defect detection
- Automated quality assessment
- Trend analysis and prediction
### Knowledge Management
- Expert experience digitalization
- Solution retrieval and recommendation
- Continuous learning and optimization
### Process Optimization
- Root cause analysis
- Performance monitoring
- Improvement suggestion generation
## Requirements
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+
- 16+ GB GPU Memory

## Training
### Yolov8
- python3.11 train_yolo.py
- python3.11 infer.py --model=./train3/weights/best.pt --image ./path/of/image.jpg --save=./out.jpg

## Installation
### Yolov8
#### web server
- YOLO_MODEL_PATH=./train_result/weights/best.pt uvicorn web_yolo:app --port 8000
#### docker
- build: ./build.sh
- run with default path:
  
  docker run -it --name yoloweb -v /home/xxx/train_result/weights/best.pt:/mnt/models/best.pt -p 8000:8000 yolov8:1.0
- run with env and path:
  
  docker run -it --name yoloweb -e YOLO_MODEL_PATH=/app/best.pt -v /home/slc/industrial-vision-intelligence/train_result/weights/best.pt:/app/best.pt -p 8000:8000 yolov8:1.0
#### python client
- python3.11 client_yolo.py -i dataset/train/images/train_1051.jpg -o out.jpg
#### http client
```
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@dataset/train/images/train_1051.jpg"
```
