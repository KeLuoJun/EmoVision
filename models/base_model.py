# models/base_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import os
from config import Config

class BaseModel(ABC, nn.Module):
    """所有模型的统一基类 - 简化版本"""
    
    def __init__(self, config: Union[Config, Dict[str, Any]]):
        super().__init__()
        self.config = self._normalize_config(config)
        self.model_name = self.__class__.__name__
        self.setup_model()
        self.setup_optimizer()
        
        # 训练历史记录
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
    
    @abstractmethod
    def setup_model(self):
        """子类必须实现：搭建模型结构"""
        pass
    
    def setup_optimizer(self):
        """设置优化器 - 支持动态调整"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        return self.model(x)
    
    def _normalize_config(self, config_input: Union[Config, Dict[str, Any]]) -> Dict[str, Any]:
        """确保内部配置统一为标准字典。"""
        if isinstance(config_input, Config):
            return asdict(config_input)
        return dict(config_input)

    def update_config(self, new_config: Union[Config, Dict[str, Any]]):
        """Apply config overrides and only rebuild optimizer when needed."""
        overrides = self._normalize_config(new_config)
        changed: Dict[str, Tuple[Any, Any]] = {}

        for key, value in overrides.items():
            previous = self.config.get(key)
            if previous != value:
                self.config[key] = value
                changed[key] = (previous, value)

        if not changed:
            print("⚠️ 配置未变化，保持现有设置。")
            return

        optimizer_sensitive = {"optimizer", "learning_rate", "weight_decay", "momentum", "betas"}
        if optimizer_sensitive.intersection(changed.keys()):
            self.setup_optimizer()

        readable_changes = {key: new for key, (_, new) in changed.items()}
        print(f"✅ 配置已更新: {readable_changes}")
    
    def get_hyperparams(self) -> Dict:
        """获取当前超参数"""
        return {
            'model': self.model_name,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.config.get('learning_rate'),
            'batch_size': self.config.get('batch_size'),
            'weight_decay': self.config.get('weight_decay', 0),
        }
    
    def save_experiment(self, experiment_name: str):
        """保存实验记录"""
        exp_data = {
            'model_name': self.model_name,
            'hyperparams': self.get_hyperparams(),
            'history': self.history,
            'final_val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else 0
        }
        
        # 保存到experiments文件夹
        os.makedirs('experiments', exist_ok=True)
        filename = f"experiments/{experiment_name}_{self.model_name}.json"
        with open(filename, 'w') as f:
            json.dump(exp_data, f, indent=2)
        print(f"💾 实验已保存: {filename}")