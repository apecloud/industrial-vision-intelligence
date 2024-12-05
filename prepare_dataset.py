import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset_structure(base_path):
    """创建数据集目录结构"""
    dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for dir in dirs:
        os.makedirs(os.path.join(base_path, dir), exist_ok=True)
    
    return base_path

def mask_to_yolo_format(mask_path, class_id, is_free_class=False):
    """将mask转换为YOLO格式的标注"""
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"Warning: Could not read mask file: {mask_path}")
        return None
        
    h, w = mask.shape
    
    if is_free_class:
        # 对于完好样本（MT_Free），使用整个图像区域作为标注
        return f"{class_id} 0 0 1 1"
    
    # 找到缺陷区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Warning: No contours found in mask: {mask_path}")
        return None
    
    # 获取最大轮廓的边界框
    cnt = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    
    # 转换为YOLO格式: <class> <x_center> <y_center> <width> <height>
    x_center = (x + w_box/2) / w
    y_center = (y + h_box/2) / h
    width = w_box / w
    height = h_box / h
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def organize_dataset(source_dirs, output_base_path, test_size=0.2):
    """组织数据集并生成标注"""
    # 检查源目录是否存在
    for dir_path in source_dirs:
        if not os.path.exists(dir_path):
            print(f"Error: Directory does not exist: {dir_path}")
            return
    
    # 创建数据集目录
    dataset_path = create_dataset_structure(output_base_path)
    
    # 类别映射
    class_mapping = {
        'MT_Blowhole': 0,
        'MT_Crack': 1,
        'MT_Break': 2,
        'MT_Fray': 3,
        'MT_Uneven': 4,
        'MT_Free': 5
    }
    
    # 收集所有数据
    all_data = []
    class_counts = {name: 0 for name in class_mapping.keys()}
    
    for dir_name in source_dirs:
        base_dir_name = os.path.basename(dir_name)
        class_id = class_mapping[base_dir_name]
        is_free_class = (base_dir_name == 'MT_Free')
        
        print(f"\nProcessing directory: {dir_name}")
        
        # 获取所有jpg文件
        jpg_files = [f for f in os.listdir(dir_name) if f.endswith('.jpg')]
        print(f"Found {len(jpg_files)} jpg files")
        
        for jpg_file in jpg_files:
            base_name = os.path.splitext(jpg_file)[0]
            jpg_path = os.path.join(dir_name, jpg_file)
            png_path = os.path.join(dir_name, f"{base_name}.png")
            
            if os.path.exists(png_path):
                all_data.append((jpg_path, png_path, class_id, is_free_class))
                class_counts[base_dir_name] += 1
            else:
                print(f"Warning: Missing PNG file for {jpg_file}")
    
    print("\nSamples per class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    
    print(f"\nTotal valid samples found: {len(all_data)}")
    if len(all_data) == 0:
        print("Error: No valid data pairs found!")
        return
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=42, 
                                          stratify=[x[2] for x in all_data])
    
    # 处理训练集
    print("\nProcessing training set...")
    successful_train = 0
    for idx, (jpg_path, png_path, class_id, is_free_class) in enumerate(train_data):
        # 复制图片
        new_jpg_name = f"train_{idx}.jpg"
        shutil.copy(jpg_path, os.path.join(dataset_path, 'train/images', new_jpg_name))
        
        # 生成标注
        yolo_annotation = mask_to_yolo_format(png_path, class_id, is_free_class)
        if yolo_annotation:
            with open(os.path.join(dataset_path, 'train/labels', f"train_{idx}.txt"), 'w') as f:
                f.write(yolo_annotation)
            successful_train += 1
    
    # 处理验证集
    print("\nProcessing validation set...")
    successful_val = 0
    for idx, (jpg_path, png_path, class_id, is_free_class) in enumerate(val_data):
        # 复制图片
        new_jpg_name = f"val_{idx}.jpg"
        shutil.copy(jpg_path, os.path.join(dataset_path, 'val/images', new_jpg_name))
        
        # 生成标注
        yolo_annotation = mask_to_yolo_format(png_path, class_id, is_free_class)
        if yolo_annotation:
            with open(os.path.join(dataset_path, 'val/labels', f"val_{idx}.txt"), 'w') as f:
                f.write(yolo_annotation)
            successful_val += 1
    
    # 生成data.yaml
    yaml_content = f"""
path: {dataset_path}
train: train/images
val: val/images

nc: 6
names: ['blowhole', 'crack', 'break', 'fray', 'uneven', 'free']
    """
    
    with open(os.path.join(dataset_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"\nDataset organized at: {dataset_path}")
    print(f"Successfully processed training samples: {successful_train}/{len(train_data)}")
    print(f"Successfully processed validation samples: {successful_val}/{len(val_data)}")

# 使用示例
if __name__ == "__main__":
    # 源数据目录列表
    source_dirs = [
        './MT_Blowhole',
        './MT_Break',
        './MT_Crack',
        './MT_Fray',
        './MT_Free',
        './MT_Uneven'
    ]
    
    # 输出目录
    output_path = './dataset'
    
    # 组织数据集
    organize_dataset(source_dirs, output_path)
