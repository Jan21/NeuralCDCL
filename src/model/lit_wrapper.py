import torch
import lightning as L
from litgpt import LLM


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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["labels"]
        logits = self._model(x)
        loss = self._loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._cfg.train.optimizer.lr)