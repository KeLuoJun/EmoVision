"""训练流程封装：统一管理训练、验证与调度逻辑。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.base_model import BaseModel


class Trainer:
    """面向 BaseModel 的轻量级训练控制器。"""

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or model.config
        self.device = torch.device(device or self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.log_interval = self.config.get("log_interval", 20)
        self.grad_accum_steps = self.config.get("grad_accum_steps", 1)
        self.scheduler = self._build_scheduler()

        self.best_val_acc: float = 0.0

    # ------------------------------------------------------------------
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Iterable[float]]:
        """运行完整训练流程并返回模型历史。"""
        epochs = num_epochs or self.config.get("epochs", 1)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                self.train_loader, train=True, epoch_idx=epoch)
            val_loss, val_acc = (0.0, 0.0)
            if self.val_loader is not None:
                val_loss, val_acc = self._run_epoch(
                    self.val_loader, train=False)
                self.best_val_acc = max(self.best_val_acc, val_acc)

            self.model.history["train_loss"].append(train_loss)
            self.model.history["train_acc"].append(train_acc)
            if self.val_loader is not None:
                self.model.history["val_loss"].append(val_loss)
                self.model.history["val_acc"].append(val_acc)

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    metric = val_loss if self.val_loader is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
        return self.model.history

    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool = True,
        epoch_idx: Optional[int] = None,
    ) -> Tuple[float, float]:
        """单轮训练或验证，返回平均损失与准确率。"""
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        optimizer = self.model.optimizer
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            inputs, targets = self._parse_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(inputs)
                raw_loss = self.criterion(outputs, targets)
                loss = raw_loss / self.grad_accum_steps if train else raw_loss

            if train:
                loss.backward()
                if step % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.get("grad_clip", float("inf")))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += raw_loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if train and step % self.log_interval == 0:
                print(
                    f"  [Epoch {epoch_idx}][{step}/{len(loader)}] "
                    f"loss={total_loss / step:.4f} acc={correct / total:.2%}"
                )

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    def _parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 DataLoader 输出统一拆分为 (inputs, targets)。"""
        if isinstance(batch, dict):
            return batch["inputs"], batch["labels"]
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("DataLoader batch 必须提供输入与标签。")

    # ------------------------------------------------------------------
    def _build_scheduler(self):
        """依据配置创建可选的学习率调度器。"""
        scheduler_name = (self.config.get("scheduler") or "").lower()
        params = self.config.get("scheduler_params", {})
        if scheduler_name == "step":
            step_size = params.get("step_size", 10)
            gamma = params.get("gamma", 0.1)
            return lr_scheduler.StepLR(self.model.optimizer, step_size=step_size, gamma=gamma)
        if scheduler_name == "cosine":
            t_max = params.get("t_max", self.config.get("epochs", 10))
            eta_min = params.get("eta_min", 0.0)
            return lr_scheduler.CosineAnnealingLR(self.model.optimizer, T_max=t_max, eta_min=eta_min)
        if scheduler_name == "plateau":
            patience = params.get("patience", 3)
            factor = params.get("factor", 0.5)
            return lr_scheduler.ReduceLROnPlateau(self.model.optimizer, factor=factor, patience=patience)
        return None

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """返回训练概览，便于日志或可视化。"""
        return {
            "config": dict(self.model.config),
            "history": self.model.history,
            "best_val_acc": self.best_val_acc,
        }