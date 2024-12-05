import torch
from PIL import Image
import faiss
import numpy as np
from cn_clip.clip import load_from_name
import os
import pickle
from tqdm import tqdm
from datetime import datetime
import shutil
import argparse
import logging
from typing import List, Dict, Optional, Tuple
import sys

class ImageVectorizer:
    def __init__(self,
                 index_file: str = "image_index.faiss",
                 mapping_file: str = "id_to_path.pkl",
                 backup_dir: str = "backups",
                 nlist: int = 100,
                 use_ivf: bool = True):
        """
        初始化图像向量化器

        Args:
            index_file: FAISS索引文件路径
            mapping_file: ID映射文件路径
            backup_dir: 备份目录
            nlist: IVF索引的聚类中心数量
            use_ivf: 是否使用IVF索引
        """
        # 设置日志
        self._setup_logging()

        # 初始化设备和模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        self.model, self.preprocess = load_from_name("ViT-B-16")
        self.model = self.model.to(self.device)
        self.model.eval()

        # 初始化存储相关
        self.dimension = 512
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.backup_dir = backup_dir
        self.id_to_path = {}

        # 创建备份目录
        os.makedirs(backup_dir, exist_ok=True)

        # 初始化索引
        if use_ivf:
            self.index = self._create_ivf_index(nlist)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        # 加载现有索引
        self.load_index()

    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger('ImageVectorizer')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _create_ivf_index(self, nlist: int) -> faiss.Index:
        """创建IVF索引"""
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        return index

    def safe_process_image(self, image_path: str) -> Optional[Image.Image]:
        """安全地处理图片"""
        try:
            image = Image.open(image_path)
            # 验证图片
            image.verify()
            # 重新打开图片(verify后需要重新打开)
            image = Image.open(image_path).convert('RGB')

            # 检查图片大小
            if image.size[0] * image.size[1] > 25000000:  # 25MP
                self.logger.warning(f"Image too large: {image_path}")
                return None

            return image
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return None

    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """单张图片编码"""
        image = self.safe_process_image(image_path)
        if image is None:
            return None

        try:
            image = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

            return image_features.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error encoding {image_path}: {e}")
            return None

    def batch_encode_images(self,
                          image_paths: List[str],
                          batch_size: int = 32) -> Optional[np.ndarray]:
        """批量编码图片"""
        all_features = []
        valid_paths = []

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                image = self.safe_process_image(path)
                if image is not None:
                    batch_images.append(self.preprocess(image))
                    valid_paths.append(path)

            if not batch_images:
                continue

            try:
                batch_tensor = torch.stack(batch_images).to(self.device)

                with torch.no_grad():
                    features = self.model.encode_image(batch_tensor)
                    features = features / features.norm(dim=1, keepdim=True)
                    all_features.append(features.cpu().numpy())

                # 清理GPU缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                continue

        return np.vstack(all_features) if all_features else None, valid_paths

    def process_in_chunks(self,
                         image_paths: List[str],
                         chunk_size: int = 1000):
        """分块处理大量图片"""
        for i in range(0, len(image_paths), chunk_size):
            chunk = image_paths[i:i + chunk_size]
            self.add_images(chunk)
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def add_images(self, image_paths: List[str], batch_size: int = 32):
        """添加图片到索引"""
        start_time = datetime.now()
        total = len(image_paths)
        self.logger.info(f"Starting to process {total} images")

        # 获取当前索引大小作为起始ID
        start_id = len(self.id_to_path)

        # 批量编码图片
        features, valid_paths = self.batch_encode_images(image_paths, batch_size)
        if features is None:
            return

        # 训练索引（如果是IVF索引且未训练）
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            self.logger.info("Training IVF index...")
            self.index.train(features)

        # 添加到索引
        self.index.add(features)

        # 更新映射关系
        for i, path in enumerate(valid_paths):
            self.id_to_path[start_id + i] = path

        # 保存索引和映射
        self.save_index()

        # 创建备份
        self.backup_index()

        elapsed = datetime.now() - start_time
        self.logger.info(f"Processed {len(valid_paths)} images in {elapsed}")

    def search(self,
              query_image_path: str,
              k: int = 5) -> List[Dict[str, float]]:
        """搜索相似图片"""
        query_feature = self.encode_image(query_image_path)
        if query_feature is None:
            return []

        # 设置IVF索引的搜索参数
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = min(20, self.index.nlist)

        distances, indices = self.index.search(query_feature, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.id_to_path:  # 确保索引有效
                results.append({
                    'path': self.id_to_path[idx],
                    'distance': float(dist)
                })

        return results

    def save_index(self):
        """保存索引和映射到本地"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.mapping_file, 'wb') as f:
                pickle.dump(self.id_to_path, f)
            self.logger.info("Index and mapping saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")

    def load_index(self):
        """从本地加载索引和映射"""
        if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.mapping_file, 'rb') as f:
                    self.id_to_path = pickle.load(f)
                self.logger.info("Index and mapping loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading index: {e}")

    def backup_index(self):
        """备份索引"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"index_backup_{timestamp}.faiss")
            shutil.copy2(self.index_file, backup_path)
            self.logger.info(f"Index backed up to {backup_path}")
        except Exception as e:
            self.logger.error(f"Error backing up index: {e}")

def main():
    parser = argparse.ArgumentParser(description='Image Vectorization and Search Tool')

    # 添加参数
    parser.add_argument('--mode', type=str, required=True, choices=['add', 'search'],
                      help='Operation mode: add images or search similar images')
    parser.add_argument('--image_dir', type=str,
                      help='Directory containing images to add')
    parser.add_argument('--query_image', type=str,
                      help='Query image path for search mode')
    parser.add_argument('--index_file', type=str, default='image_index.faiss',
                      help='Path to FAISS index file')
    parser.add_argument('--mapping_file', type=str, default='id_to_path.pkl',
                      help='Path to ID-path mapping file')
    parser.add_argument('--backup_dir', type=str, default='backups',
                      help='Directory for index backups')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing images')
    parser.add_argument('--chunk_size', type=int, default=1000,
                      help='Chunk size for processing large datasets')
    parser.add_argument('--top_k', type=int, default=5,
                      help='Number of similar images to return in search mode')
    parser.add_argument('--use_ivf', action='store_true',
                      help='Use IVF index for better search performance')

    args = parser.parse_args()

    # 初始化向量化器
    vectorizer = ImageVectorizer(
        index_file=args.index_file,
        mapping_file=args.mapping_file,
        backup_dir=args.backup_dir,
        use_ivf=args.use_ivf
    )

    if args.mode == 'add':
        if not args.image_dir:
            raise ValueError("--image_dir is required for add mode")

        # 获取所有图片路径
        image_paths = []
        for ext in ('.png', '.jpg', '.jpeg', '.webp'):
            image_paths.extend(
                os.path.join(args.image_dir, f)
                for f in os.listdir(args.image_dir)
                if f.lower().endswith(ext)
            )

        # 分块处理图片
        vectorizer.process_in_chunks(image_paths, args.chunk_size)

    elif args.mode == 'search':
        if not args.query_image:
            raise ValueError("--query_image is required for search mode")

        # 搜索相似图片
        results = vectorizer.search(args.query_image, args.top_k)

        # 打印结果
        print("\nSearch results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Path: {result['path']}")
            print(f"   Distance: {result['distance']:.4f}")

if __name__ == "__main__":
    main()
