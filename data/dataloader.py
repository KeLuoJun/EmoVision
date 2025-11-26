"""数据加载与 Dataset 定义。"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config
from data.transforms import get_transforms


class EmotionROIDataset(Dataset):
    """
    EmotionROI 数据集加载器。

    Args:
        root_dir: 数据集根目录 (包含 images/ 和 txt 文件)
        split_file: 划分文件名称 (如 training.txt, testing.txt)
        transform: 图像预处理转换
    """

    # 类别映射 (按字母顺序)
    CLASSES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

    def __init__(self, root_dir: str, split_file: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        split_path = os.path.join(root_dir, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # line format: class/image.jpg
                # image path is relative to images/ folder
                img_rel_path = line

                # 解析类别
                class_name = os.path.dirname(img_rel_path)
                if class_name not in self.CLASS_TO_IDX:
                    # 尝试从文件名或其他方式获取，这里假设文件夹名即类别
                    continue

                label = self.CLASS_TO_IDX[class_name]
                img_path = os.path.join(root_dir, 'images', img_rel_path)

                self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个全黑图像作为 fallback，或者抛出异常
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练和验证 DataLoader。

    Args:
        config: 全局配置对象

    Returns:
        (train_loader, val_loader)
    """
    train_transforms, val_transforms = get_transforms(config.input_size)

    # 确保数据集路径正确
    data_root = os.path.join(config.dataset_root, "EmotionROI_6")
    if not os.path.exists(data_root):
        # 回退到直接使用 dataset_root
        data_root = config.dataset_root

    train_dataset = EmotionROIDataset(
        root_dir=data_root,
        split_file="training.txt",
        transform=train_transforms
    )

    val_dataset = EmotionROIDataset(
        root_dir=data_root,
        split_file="testing.txt",
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )

    return train_loader, val_loader
