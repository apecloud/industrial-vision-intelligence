# industrial-vision-intelligence

## Introduction
IVI System is an innovative solution that combines traditional deep learning models, vision-language models, and large language models to address quality control and knowledge management challenges in industrial manufacturing. This project leverages the strengths of YOLOv8, CogVLM, Qwen-72B, and ViT-B-16 to create a comprehensive system for defect detection, analysis, and knowledge extraction.

We demonstrate the system's effectiveness using the publicly available Magnetic Tile Defect Dataset https://github.com/abin24/Magnetic-tile-defect-datasets. as a benchmark case study. The implementation workflow consists of three main components:

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
