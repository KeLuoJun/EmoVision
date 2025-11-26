"""训练结果可视化工具。"""

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns


def plot_experiment_results(results: Dict[str, Dict], save_dir: str = "experiments/charts", prefix: str = ""):
    """
    绘制多模型对比曲线并保存。

    Args:
        results: 实验结果字典，格式 {model_name: {history: {...}, ...}}
        save_dir: 图表保存目录
        prefix: 文件名前缀 (通常是实验名+时间戳)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 设置风格
    sns.set_theme(style="whitegrid")

    # 准备数据
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    titles = {
        'train_loss': 'Training Loss',
        'val_loss': 'Validation Loss',
        'train_acc': 'Training Accuracy',
        'val_acc': 'Validation Accuracy'
    }

    # 创建画布: 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Experiment Comparison: {prefix}', fontsize=16)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        has_data = False
        for model_name, data in results.items():
            history = data.get('history', {})
            if metric in history and history[metric]:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric],
                        label=model_name, marker='o', markersize=4)
                has_data = True

        ax.set_title(titles[metric])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.split('_')[1].capitalize())
        if has_data:
            ax.legend()

    plt.tight_layout()

    # 保存图片
    filename = f"{prefix}_comparison.png" if prefix else "comparison.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 可视化图表已保存: {save_path}")
