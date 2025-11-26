"""简单的 CNN 基线模型。"""

import torch.nn as nn
from models.base_model import BaseModel


class SimpleCNN(BaseModel):
    """
    一个简单的 3 层卷积神经网络，用于快速验证流程。
    结构: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC
    """

    def setup_model(self):
        num_classes = self.config.get("num_classes", 7)

        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28

            # Classifier
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
