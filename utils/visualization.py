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
        results: 实验结果字典，格式 {model_name: {history: {...}, hyperparams: {...}, ...}}
        save_dir: 图表保存目录
        prefix: 文件名前缀 (通常是实验名+时间戳)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 设置风格
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    # 使用 tab10 配色，适合分类对比
    palette = sns.color_palette("tab10", n_colors=max(len(results), 10))

    # 准备数据
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    titles = {
        'train_loss': 'Training Loss',
        'val_loss': 'Validation Loss',
        'train_acc': 'Training Accuracy',
        'val_acc': 'Validation Accuracy'
    }

    # 定义不同图表的样式
    styles = {
        'train_loss': {'linestyle': '-', 'marker': 'o'},   # 实线 + 圆点
        'val_loss':   {'linestyle': '--', 'marker': 's'},  # 虚线 + 方块
        'train_acc':  {'linestyle': '-', 'marker': '^'},   # 实线 + 三角
        'val_acc':    {'linestyle': '--', 'marker': 'D'}   # 虚线 + 菱形
    }

    # 创建画布: 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)

    # 主标题
    title_text = f'Experiment Comparison: {prefix}' if prefix else 'Experiment Comparison'
    fig.suptitle(title_text, fontsize=22,
                 fontweight='bold', y=0.96, color='#333333')

    # 收集参数信息用于显示
    param_text = []

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        style = styles.get(metric, {'linestyle': '-', 'marker': 'o'})

        has_data = False
        for i, (model_name, data) in enumerate(results.items()):
            history = data.get('history', {})
            hyperparams = data.get('hyperparams', {})

            # 仅在第一个图收集参数信息，避免重复
            if idx == 0:
                # 筛选关键参数
                key_params = {k: v for k, v in hyperparams.items(
                ) if k in ['learning_rate', 'batch_size', 'optimizer', 'weight_decay']}
                if key_params:
                    params_str = ", ".join(
                        [f"{k}={v}" for k, v in key_params.items()])
                    param_text.append(f"• {model_name}: {params_str}")

            if metric in history and history[metric]:
                values = history[metric]
                epochs = range(1, len(values) + 1)

                # 绘制曲线
                ax.plot(epochs, values,
                        label=model_name,
                        color=palette[i],
                        linewidth=2.5,
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        markersize=8,
                        markeredgecolor='white',
                        markeredgewidth=1.5,
                        alpha=0.9)

                # 标注最佳点 (Loss取最小，Acc取最大)
                if 'loss' in metric:
                    best_val = min(values)
                    best_idx = values.index(best_val)
                    offset = (0, -20)
                    va = 'top'
                else:
                    best_val = max(values)
                    best_idx = values.index(best_val)
                    offset = (0, 15)
                    va = 'bottom'

                # 添加数值标注
                ax.annotate(f'{best_val:.4f}',
                            xy=(epochs[best_idx], best_val),
                            xytext=offset, textcoords='offset points',
                            ha='center', va=va,
                            fontsize=10,
                            color=palette[i],
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9, ec=palette[i], lw=1))

                has_data = True

        ax.set_title(titles[metric], fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel(metric.split('_')[1].capitalize(), fontsize=12)

        # 美化网格
        ax.grid(True, linestyle='--', alpha=0.4)

        # 去除上边框和右边框 (Despine)
        sns.despine(ax=ax)

        if has_data:
            ax.legend(frameon=True, fancybox=True,
                      shadow=True, loc='best', fontsize=10)

    # 在图表底部添加参数信息文本框
    if param_text:
        info_text = "\n".join(param_text)
        # 使用文本框显示参数，放在底部，使用等宽字体对齐
        props = dict(boxstyle='round', facecolor='#f8f9fa',
                     alpha=0.95, edgecolor='#dee2e6', pad=1)
        fig.text(0.5, 0.02, info_text, ha='center', va='bottom',
                 fontsize=11, fontfamily='monospace', bbox=props, color='#444444')

    # 调整布局，为底部文本框留出空间
    plt.tight_layout(rect=[0, 0.12, 1, 0.94])

    # 保存图片
    filename = f"{prefix}_comparison.png" if prefix else "comparison.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 可视化图表已保存: {save_path}")
