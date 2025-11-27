# 🎭 EmoVision

> 面向图像识别研究与ML/DL视觉模型训练的实验框架  
> 导入数据与模型架构，一键开始训练 · 评估 · 可视化 · 记录  
> 统一配置 · 多模型对比 · 自动可视化

---

## ✨ 核心特性

- 🧩 **模块化架构** — `models/`、`data/`、`training/`、`utils/` 各司其职，方便替换模型与组件
- 📊 **多模型对比** — 支持一次性训练多个模型，并自动生成对比图表
- ⚙️ **集中配置** — 所有超参数由 `config.Config` 管理，支持命令行参数覆盖
- 💾 **实验留痕** — 自动保存带时间戳的实验结果（JSON）与可视化图表（PNG）
- 📋 **团队规范** — `开发规范.md` 约束命名、分支、日志等习惯，降低多人协作成本

---

## 📁 仓库结构
```
EmoVision/
├── config.py              # Config 数据类，集中管理超参数
├── models/                # 模型定义
│   ├── base_model.py      # 模型基类
│   └── simple_cnn.py      # 示例模型
├── data/                  # 数据处理
│   ├── dataloader.py      # EmotionROI 数据集加载
│   └── transforms.py      # 数据增强策略
├── training/              # 训练核心
│   └── trainer.py         # 训练循环封装
├── utils/                 # 工具库
│   └── visualization.py   # 结果可视化绘制
├── experiments/           # 实验产物
│   ├── *.json             # 训练指标记录（带时间戳）
│   └── charts/            # 自动生成的对比图表
├── main.py                # 主程序（支持 CLI）
└── 开发规范.md             # 团队开发约定
```

---

## 🚀 快速开始速开始

### 1️⃣ 环境准备
```bash
pip install -r requirements.txt
```

### 2️⃣ 数据格式与放置
- 默认支持 **EmotionROI** 数据集。
- 请将数据解压至 `data/data_name/`，结构如下：
  - `images/` (包含不同类别的图片)
  - `training.txt` （每张图片的路径、标签信息, eg. `anger/234.jpg`）
  - `testing.txt` （每张图片的路径、标签信息, eg. `anger/234.jpg`）


## 3️⃣ 添加自定义模型

### 1. 定义模型类
在 `models/` 下新建文件（例如 `my_new_model.py`），继承 `BaseModel` 并实现 `setup_model` 方法：

```python
from .base_model import BaseModel
import torch.nn as nn

class MyNewModel(BaseModel):
    def setup_model(self):
        # 使用 self.config 访问配置
        self.net = nn.Sequential(
            nn.Linear(self.config.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.num_classes)
        )
        # ⚠️ 必须将核心网络赋值给 self.model，Trainer 会自动将其移至 GPU
        self.model = self.net
```

### 2. 注册模型
打开 `main.py`，导入你的新模型类并加入 `MODEL_ZOO` 字典：

```python
# main.py
from models.my_new_model import MyNewModel  # <--- 导入

MODEL_ZOO = {
    "SimpleCNN": SimpleCNN,
    "MyNewModel": MyNewModel,  # <--- 注册
}
```

### 3. 运行
```bash
python main.py --models MyNewModel
```

## 4️⃣ 运行训练
**基础运行（默认 SimpleCNN）：**
```bash
python main.py
```

**高级用法（多模型对比 + 参数覆盖）：**
```bash
# 运行指定模型，修改 epoch 和 batch size，指定实验名
python main.py --models SimpleCNN --epochs 20 --batch_size 16 --exp_name my_test
```

> 💡 **支持参数**：`--models` (列表), `--epochs`, `--batch_size`, `--lr`, `--exp_name`

---

## 📈 训练流程约定
1. 在 `training/trainer.py` 中实现标准化训练循环。
2. 所有评估指标写在 `training/metrics.py`，并附简短用法注释。
3. 可选回调（早停、日志推送等）统一放入 `training/callbacks.py`

---

## 📝 实验记录验记录

- 📄 **数据记录** — 训练结束后，会在 `experiments/training_history` 下生成 `{exp_name}_{timestamp}.json`，包含所有运行模型的超参、Loss/Acc 历史与最终结果
- 📊 **可视化** — 在 `experiments/charts/` 下生成对比图表（Loss 曲线、Accuracy 曲线、混淆矩阵），方便直观评估
- 💾 **模型保存** — 在 `experiments/trained_models/`下保存训练好的模型权重，便于后续加载与推理
- ✍️ **手动记录** — 推荐配合 Markdown 文档记录实验结论，引用自动生成的图表

---

## 📖 开发规范发规范

完整说明见 [`开发规范.md`](./开发规范.md)，主要包括：

- ✅ 代码与命名风格、类型注解约定
- ✅ Config 使用方式、模型继承要求
- ✅ 数据/训练模块职责划分与文件上限
- ✅ Git 分支、提交信息与自检流程

