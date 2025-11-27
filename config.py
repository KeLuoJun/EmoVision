"""全局配置文件：集中管理模型训练所需的关键超参数。"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch


@dataclass
class Config:
    """仅存放超参数属性，方便在项目中统一调用。"""

    # ===== 实验基础设置 =====
    experiment_name: str = "baseline"  # 实验名称，用于区分不同配置
    seed: int = 42  # 随机种子，保证实验可复现
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备

    # ===== 数据相关 =====
    dataset_root: str = "data"  # 数据集根目录
    dataset_name: str = "EmotionROI_6"  # 数据集名称
    num_classes: int = 7  # 分类类别数量
    input_size: int = 224  # 输入图像尺寸
    num_workers: int = 4  # DataLoader 的并行加载进程数

    # ===== 训练超参数 =====
    epochs: int = 5  # 训练轮数
    batch_size: int = 32  # 每批数据大小
    learning_rate: float = 1e-3  # 学习率
    weight_decay: float = 1e-4  # 权重衰减
    optimizer: str = "adam"  # 优化器名称，可选 adam/sgd/adamw
    momentum: float = 0.9  # SGD 优化器用到的动量
    betas: Optional[tuple] = (0.9, 0.999)  # Adam 系列优化器的 beta 参数

    # ===== 学习率调度与正则化 =====
    scheduler: Optional[str] = None  # 学习率调度器名称，暂不启用可设为 None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)  # 调度器参数字典

    # ===== 模型保存与日志 =====
    checkpoint_dir: str = "experiments"  # 模型与日志输出目录
