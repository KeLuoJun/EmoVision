"""EmoVision 主程序入口。"""

import argparse
import json
import os
import sys
from datetime import datetime
import torch

from config import Config
from data.dataloader import get_dataloaders
from models.simple_cnn import SimpleCNN
from training.trainer import Trainer
from utils.visualization import plot_experiment_results, plot_confusion_matrix


# 模型注册表
MODEL_ZOO = {
    "SimpleCNN": SimpleCNN
}


def parse_args():
    parser = argparse.ArgumentParser(description="EmoVision Training")
    parser.add_argument("--models", type=str, nargs="+", default=["SimpleCNN"],
                        help=f"Models to train (space separated). Available: {list(MODEL_ZOO.keys())}")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--batch_size", type=int,
                        default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--exp_name", type=str,
                        default=None, help="Experiment name")
    return parser.parse_args()


def main():
    # 1. 初始化配置
    args = parse_args()
    cfg = Config()

    # 命令行参数覆盖默认配置
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.exp_name:
        overrides["experiment_name"] = args.exp_name

    # 2. 准备数据
    print("📦 正在加载数据...")
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"   训练集大小: {len(train_loader.dataset)}")
    print(f"   验证集大小: {len(val_loader.dataset)}")

    # 准备保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = overrides.get("experiment_name", cfg.experiment_name)
    charts_dir = os.path.join("experiments/charts", f"{exp_name}_{timestamp}")
    os.makedirs(charts_dir, exist_ok=True)

    # 3. 遍历训练选定的模型
    experiment_results = {}

    for model_name in args.models:
        if model_name not in MODEL_ZOO:
            print(f"⚠️  跳过未知模型: {model_name} (可用: {list(MODEL_ZOO.keys())})")
            continue

        print(f"\n{'='*20} 正在初始化: {model_name} {'='*20}")
        ModelClass = MODEL_ZOO[model_name]

        # 每次实例化一个新的模型对象
        model = ModelClass(cfg)

        # 应用命令行覆盖参数
        if overrides:
            model.update_config(overrides)

        # 4. 初始化训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # 5. 开始训练
        print(f"🚀 开始训练: {model.config['experiment_name']} ({model_name})")
        print(f"   设备: {trainer.device}")
        print("-" * 60)

        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n🛑 训练被用户中断，保存当前进度...")

        # 保存训练好的模型权重
        # timestamp 使用循环外定义的统一时间戳
        save_dir = "experiments/trained_models"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(
            save_dir, f"{model_name}_{timestamp}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 模型权重已保存至: {model_save_path}")

        # 获取预测结果并绘制混淆矩阵
        print(f"📊 正在生成 {model_name} 的混淆矩阵...")
        y_true, y_pred = trainer.get_predictions()

        # 获取类别名称
        idx_to_class = {v: k for k,
                        v in train_loader.dataset.class_to_idx.items()}
        classes = [idx_to_class[i] for i in range(len(idx_to_class))]

        plot_confusion_matrix(y_true, y_pred, classes, charts_dir, model_name)

        # 6. 收集实验结果
        print("-" * 60)
        experiment_results[model_name] = {
            'model_name': model_name,
            'hyperparams': model.get_hyperparams(),
            'history': model.history,
            'final_val_acc': model.history['val_acc'][-1] if model.history['val_acc'] else 0,
            'model_path': model_save_path
        }
        print(f"✨ 模型 {model_name} 训练结束!")

    # 7. 统一保存所有结果
    if experiment_results:
        # exp_name 和 timestamp 已经在上面定义

        os.makedirs("experiments/training_history", exist_ok=True)
        save_path = os.path.join(
            "experiments/training_history", f"{exp_name}_{timestamp}.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 所有模型训练记录已保存至: {save_path}")

        # 8. 生成可视化图表
        try:
            chart_prefix = f"{exp_name}_{timestamp}"
            plot_experiment_results(
                experiment_results, save_dir=charts_dir, prefix=chart_prefix)
        except Exception as e:
            print(f"⚠️  可视化生成失败: {e}")

    print("\n🎉 所有任务已结束。")


if __name__ == "__main__":
    main()
