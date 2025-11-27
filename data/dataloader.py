"""数据加载与 Dataset 定义。"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config
from data.transforms import get_transforms


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    自动扫描目录下的子文件夹作为类别。

    Args:
        directory: 包含类别子文件夹的根目录

    Returns:
        (classes, class_to_idx)
    """
    if not os.path.exists(directory):
        return [], {}

    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    if not classes:
        return [], {}

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class GenericImageDataset(Dataset):
    """
    通用的图像分类数据集加载器。
    支持通过 txt 文件指定划分，或（未来可扩展）直接遍历目录。

    Args:
        root_dir: 数据集根目录 (包含 images/ 和 txt 文件)
        split_file: 划分文件名称 (如 training.txt, testing.txt)
        class_to_idx: 类别名到索引的映射字典
        transform: 图像预处理转换
    """

    def __init__(self, root_dir: str, split_file: str, class_to_idx: Dict[str, int], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []

        split_path = os.path.join(root_dir, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        # 预先检查 images 目录
        self.images_dir = os.path.join(root_dir, 'images')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}")

        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # line format: class/image.jpg
                img_rel_path = line

                # 解析类别：优先使用路径中的父目录名
                class_name = os.path.dirname(img_rel_path)

                if class_name not in self.class_to_idx:
                    continue

                label = self.class_to_idx[class_name]
                img_path = os.path.join(self.images_dir, img_rel_path)

                self.samples.append((img_path, label))

        print(f"   Loaded {len(self.samples)} samples from {split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练和验证 DataLoader。
    自动检测类别数量并更新 config。

    Args:
        config: 全局配置对象

    Returns:
        (train_loader, val_loader)
    """
    train_transforms, val_transforms = get_transforms(config.input_size)

    # 1. 确定数据集路径
    data_root = os.path.join(config.dataset_root, config.dataset_name)
    if not os.path.exists(data_root):
        # 回退尝试
        if os.path.exists(os.path.join(config.dataset_root, "images")):
            data_root = config.dataset_root
        else:
            raise FileNotFoundError(
                f"Dataset not found at {data_root} or {config.dataset_root}")

    print(f"📂 Dataset root: {data_root}")

    # 2. 自动检测类别
    images_dir = os.path.join(data_root, 'images')
    classes, class_to_idx = find_classes(images_dir)

    if not classes:
        raise ValueError(
            f"No classes found in {images_dir}. Ensure structure is data/dataset/images/class_name/")

    num_classes = len(classes)
    print(f"🔍 Found {num_classes} classes: {classes}")

    # 3. 更新 Config 中的类别数
    config.num_classes = num_classes

    # 4. 构建数据集
    train_dataset = GenericImageDataset(
        root_dir=data_root,
        split_file="training.txt",
        class_to_idx=class_to_idx,
        transform=train_transforms
    )

    val_dataset = GenericImageDataset(
        root_dir=data_root,
        split_file="testing.txt",
        class_to_idx=class_to_idx,
        transform=val_transforms
    )

    # 5. 构建 DataLoader
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
