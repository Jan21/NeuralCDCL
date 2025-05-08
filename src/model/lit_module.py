import torch
from torch.utils.data import WeightedRandomSampler
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from litgpt import LLM
import math
from src.dataset.dataset import TokenizedDataset
from src.dataset.dataloader_builder import DataloaderBuilder
from omegaconf import DictConfig
import numpy as np


def linear_warmup_then_cosine(warmup_steps: int, total_steps: int, min_lr: float, peak_lr: float):
    def fn(step: int):
        if step < warmup_steps:
            return (float(step) / float(max(1, warmup_steps)))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = min_lr + (peak_lr - min_lr) * cosine
        return lr / peak_lr  # LambdaLR expects multiplier of initial LR
    return fn


class LitTrainingModule(L.LightningModule):
    def __init__(
        self, model: LLM, cfg: DictConfig, total_steps: int, datasets: list[TokenizedDataset], 
        dataloader_builder: DataloaderBuilder
    ):
        super().__init__()
        self._model = model
        self._cfg = cfg
        self._block_size = cfg.train.model.block_size
        self._max_epochs = cfg.train.trainer.epochs
        self._curriculum_temp = cfg.train.dataset.curriculum_temp
        self._datasets = datasets
        self._dataloader_builder = dataloader_builder
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._total_steps = total_steps
        self.save_hyperparameters()

    def forward(self, x):
        return self._model.forward(x)

    def training_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        x, y = batch["input_ids"], batch["labels"]
        logits = self._model(x)
        loss = self._loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def on_after_backward(self):
        if self.global_step % 10 == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.log("grad_norm", total_norm, prog_bar=False, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        x, y = batch["input_ids"], batch["labels"]
        logits = self._model(x)
        loss = self._loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def train_dataloader(self):
        epoch = self.trainer.current_epoch

        weights = self._datasets['train'].get_curriculum_weights(epoch, self._max_epochs, temp=self._curriculum_temp)
        lengths = self._datasets['train'].get_lengths()
        lengths_np = np.array(lengths, dtype=np.float32)
        weights_np = weights.numpy()

        weighted_mean = float((weights_np * lengths_np).sum())
        self.logger.experiment.log({"curriculum/length_weighted_mean": weighted_mean, "epoch": self.trainer.current_epoch})

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return self._dataloader_builder.build_dataloader(self._datasets['train'], sampler=sampler)

    def val_dataloader(self):
        return self._dataloader_builder.build_dataloader(
            self._datasets['val'],
            shuffle=False
        )

    def configure_optimizers(self):
        cfg = self._cfg.train

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.optimizer.peak_lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas),
        )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=linear_warmup_then_cosine(
                warmup_steps=cfg.optimizer.warmup_steps,
                total_steps=self._total_steps,
                min_lr=cfg.optimizer.min_lr,
                peak_lr=cfg.optimizer.peak_lr
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
