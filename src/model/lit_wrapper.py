import torch
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from litgpt import LLM


def linear_warmup_then_const(warmup_steps: int):
    def fn(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0  # stay constant
    return fn


class LitWrapper(L.LightningModule):
    def __init__(self, model: LLM, cfg):
        super().__init__()
        self._model = model
        self.block_size = cfg.train.model.block_size
        self._cfg = cfg
        self._loss_fn = torch.nn.CrossEntropyLoss()
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
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas),
        )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=linear_warmup_then_const(cfg.optimizer.warmup_steps)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
