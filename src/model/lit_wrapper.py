import torch
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from litgpt import LLM
import math


def linear_warmup_then_cosine(warmup_steps: int, total_steps: int, min_lr: float, peak_lr: float):
    def fn(step: int):
        if step < warmup_steps:
            return (float(step) / float(max(1, warmup_steps))) * (peak_lr / peak_lr)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = min_lr + (peak_lr - min_lr) * cosine
        return lr / peak_lr  # LambdaLR expects multiplier of initial LR
    return fn


class LitWrapper(L.LightningModule):
    def __init__(self, model: LLM, cfg, total_steps: int):
        super().__init__()
        self._model = model
        self.block_size = cfg.train.model.block_size
        self._cfg = cfg
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._total_steps = total_steps
        self.save_hyperparameters()

    def forward(self, x):
        return self._model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["labels"]
        logits = self._model(x)
        loss = self._loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["labels"]
        logits = self._model(x)
        loss = self._loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

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
