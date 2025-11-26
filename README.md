# EmoVision

面向情感识别研究与落地场景的视觉模型实验框架。

## 核心特性
- **模块化架构**：`models/`、`data/`、`training/`、`utils/` 各司其职，方便替换模型与组件。
- **集中配置**：所有超参数由 `config.Config` 管理，可在运行期安全地 `update_config`。
- **实验留痕**：内置 `BaseModel.save_experiment` 与 `experiments/` 目录，保证每次训练都有记录可追溯。
- **团队规范**：`开发规范.md` 约束命名、分支、日志等习惯，降低多人协作成本。

## 仓库结构
```
EmoVision/
├── config.py              # Config 数据类，集中管理超参数
├── models/                # 模型定义（BaseModel + 各模型）
├── data/                  # Dataset、DataLoader 与增强策略
├── training/              # Trainer、Metrics、Callbacks
├── utils/                 # 可视化、通用工具
├── experiments/           # 实验日志与产物
├── main.py                # 训练/评估入口
└── 开发规范.md             # 团队开发约定
```

## 快速开始
### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据放置
- 默认数据根目录为 `data/`，可在 `Config.dataset_root` 中修改。
- 推荐结构：`data/{train,val,test}/{class_name}/*.jpg`，具体取决于自定义 `Dataset`。

### 3. 运行训练
```bash
python main.py
```

## 配置与模型示例
```python
from config import Config
from models.your_model import YourModel

cfg = Config(learning_rate=5e-4, batch_size=64)
model = YourModel(cfg)

# 在线调整部分超参数
model.update_config({
	"scheduler": "cosine",
	"scheduler_params": {"t_max": 30}
})
```
- `Config` 仅包含属性，所有字段已在 `config.py` 中标注中文注释。
- `update_config` 只会在必要时重建优化器，避免无谓开销。

## 训练流程约定
1. 在 `training/trainer.py` 中实现标准化训练循环。
2. 所有评估指标写在 `training/metrics.py`，并附简短用法注释。
3. 可选回调（早停、日志推送等）统一放入 `training/callbacks.py`。

## 实验记录
- 训练结束后调用 `BaseModel.save_experiment` 会在 `experiments/` 下生成 JSON。
- 推荐同时撰写 Markdown：命名格式 `YYYY-MM-DD-Model-Remark.md`，附配置、指标、结论与 TODO。
- 可视化图表（Loss/Acc、混淆矩阵）置于 `experiments/assets/`，在文档中相对引用。

## 开发规范
完整说明见 `开发规范.md`，主要包括：
- 代码与命名风格、类型注解约定。
- Config 使用方式、模型继承要求。
- 数据/训练模块职责划分与文件上限。
- Git 分支、提交信息与自检流程。

