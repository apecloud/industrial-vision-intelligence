import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import argparse

class Settings(BaseSettings):
    # API 配置
    api_host: str = "localhost"
    api_port: int = 8000
    api_endpoint: str = "detect"
    
    # 字体配置
    font_path: Optional[str] = None
    font_size: int = 20
    
    # 显示配置
    show_result: bool = False
    
    class Config:
        env_prefix = "APP_"

    def get_api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}/{self.api_endpoint}"

settings = Settings()

def parse_args():
    parser = argparse.ArgumentParser(description='Defect Detection Tool')
    
    # 输入相关参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--image', type=Path,
                            help='单个图片的路径')
    input_group.add_argument('-d', '--directory', type=Path,
                            help='要处理的图片目录')
    
    # 输出相关参数
    parser.add_argument('-o', '--output', type=Path,
                            help='输出路径（单图片模式为文件路径，目录模式为目录路径）')
    
    # 其他选项
    parser.add_argument('--use-cv2', action='store_true', default=True,
                       help='使用OpenCV而不是PIL进行绘制 (默认: True)')
    parser.add_argument('--show', action='store_true',
                       help='显示处理结果')
    parser.add_argument('--font', type=str,
                       help='字体文件路径（用于PIL模式）')
    
    args = parser.parse_args()
    
    # 验证输入路径
    if args.image and not args.image.exists():
        parser.error(f"输入图片不存在: {args.image}")
    if args.directory and not args.directory.exists():
        parser.error(f"输入目录不存在: {args.directory}")
        
    # 创建输出目录（如果需要）
    if args.output:
        if args.directory:  # 目录模式
            args.output.mkdir(parents=True, exist_ok=True)
        else:  # 单图片模式
            args.output.parent.mkdir(parents=True, exist_ok=True)
            
    return args

def draw_detections_cv2(image_path: Path, detections: list, output_path: Optional[Path] = None):
    """使用 OpenCV 绘制检测结果"""
    image = cv2.imread(str(image_path))
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = float(det['confidence'])
        class_name = det['class']
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f'{class_name}: {conf:.2f}'
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(image, (x1, y1 - label_height - baseline - 5),
                     (x1 + label_width, y1), (0, 255, 0), -1)
        
        cv2.putText(image, label, (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if output_path:
        cv2.imwrite(str(output_path), image)
        
    if settings.show_result:
        cv2.imshow('Detection Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return image

def draw_detections_pil(image_path: Path, detections: list, output_path: Optional[Path] = None):
    """使用 PIL 绘制检测结果"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(settings.font_path, settings.font_size) if settings.font_path else ImageFont.load_default()
    except Exception as e:
        print(f"Font loading error: {e}")
        font = ImageFont.load_default()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = float(det['confidence'])
        class_name = det['class']
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
        
        label = f'{class_name}: {conf:.2f}'
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([(x1, y1 - text_height - 4), (x1 + text_width, y1)],
                      fill=(0, 255, 0))
        
        draw.text((x1, y1 - text_height - 4), label, fill=(0, 0, 0), font=font)

    if output_path:
        image.save(output_path)
    return image

def detect_defects(image_path: Path, output_path: Optional[Path] = None, use_cv2: bool = True):
    """检测缺陷并绘制结果"""
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(settings.get_api_url(), files=files)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    results = response.json()
    
    if results.get('detections'):
        if use_cv2:
            processed_image = draw_detections_cv2(image_path, results['detections'], output_path)
        else:
            processed_image = draw_detections_pil(image_path, results['detections'], output_path)
        return results, processed_image
    
    return results, None

def process_directory(input_dir: Path, output_dir: Optional[Path] = None, use_cv2: bool = True):
    """处理整个目录的图片"""
    results = []
    for image_path in input_dir.glob("*.[jp][pn][g]"):  # 匹配 jpg, jpeg, png
        try:
            output_path = None
            if output_dir:
                output_path = output_dir / f"processed_{image_path.name}"
                
            result, _ = detect_defects(image_path, output_path, use_cv2)
            results.append({"image": image_path.name, "result": result})
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({"image": image_path.name, "error": str(e)})
    
    return results

def main():
    args = parse_args()
    
    # 更新全局设置
    settings.show_result = args.show
    if args.font:
        settings.font_path = args.font
    
    try:
        if args.image:  # 处理单张图片
            results, _ = detect_defects(
                args.image,
                args.output,
                use_cv2=args.use_cv2
            )
            print(f"Results for {args.image}:")
            print(results)
            
        else:  # 处理目录
            results = process_directory(
                args.directory,
                args.output,
                use_cv2=args.use_cv2
            )
            print("Directory processing results:")
            for result in results:
                print(f"\n{result['image']}:")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(result['result'])
                    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
