"""数据增强与预处理策略。"""

from typing import Tuple

import torch
from torchvision import transforms


def get_transforms(input_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    获取训练和验证的转换管道。

    Args:
        input_size: 输入图像的大小

    Returns:
        (train_transforms, val_transforms)
    """
    # 基础标准化参数 (ImageNet 均值方差)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transforms, val_transforms
