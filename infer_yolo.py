import torch
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import signal
import sys

class Detector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = ['blowhole', 'crack', 'break', 'fray', 'uneven', 'free']
        
    def detect(self, image):
        results = self.model(image,
                           conf=self.conf_thres,
                           iou=self.iou_thres)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': cls,
                    'class_name': self.class_names[cls]
                })
        
        return detections
    
    def draw_results(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['conf']:.2f}"
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

def signal_handler(sig, frame):
    print('按下Ctrl+C，程序退出')
    cv2.destroyAllWindows()
    sys.exit(0)

def process_image(image_path, model_path, save_path=None):
    # 设置Ctrl+C信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 初始化检测器
    detector = Detector(model_path)
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # 图片检测
    detections = detector.detect(image)
    
    # 绘制结果
    result_image = detector.draw_results(image.copy(), detections)
    
    # 保存结果
    if save_path:
        cv2.imwrite(save_path, result_image)
    
    # 显示结果
    cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Detection Result', result_image)
    
    print(f"检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections):
        print(f"目标 {i+1}:")
        print(f"  类别: {det['class_name']} (id: {det['class']})")
        print(f"  置信度: {det['conf']:.2f}")
        print(f"  边界框: {[round(x) for x in det['bbox']]}")
    
    # 等待Ctrl+C
    try:
        while True:
            if cv2.waitKey(100) != -1:  # 每100ms检查一次键盘输入
                break
    except KeyboardInterrupt:
        print('按下Ctrl+C，程序退出')
    finally:
        cv2.destroyAllWindows()
    
    return result_image, detections

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8目标检测')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--image', type=str, required=True,
                        help='待检测图片路径')
    parser.add_argument('--save', type=str, default=None,
                        help='结果保存路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS的IOU阈值')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 处理图片
        detector = Detector(
            model_path=args.model,
            conf_thres=args.conf,
            iou_thres=args.iou
        )
        
        result_image, detections = process_image(
            image_path=args.image,
            model_path=args.model,
            save_path=args.save
        )
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return

if __name__ == "__main__":
    main()
