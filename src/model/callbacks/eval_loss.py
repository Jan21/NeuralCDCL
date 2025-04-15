import torch
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from typing import Callable
from typing import Optional


class EvalLossCallback(Callback):
    def __init__(
        self,
        loader: DataLoader,
        loader_name: str,
        eval_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
        loss_name: str,
        sample_size: Optional[int] = None,
        eval_every_n_epochs: int = 1
    ):
        """
        Args:
            loader: The dataloader to use for evaluation.
            loader_name: The dataloader's name.
            eval_fn: A function that takes (pl_module, batch) and returns a scalar torch.Tensor loss.
            eval_every_n_epochs: Frequency of evaluation.
            loss_name: Name for logging.
            sample_size: Number of examples to evaluate per trace type.
        """
        self._loader = loader
        self._loader_name = loader_name
        self._eval_fn = eval_fn
        self._loss_name = loss_name
        self._sample_size = sample_size
        self._eval_every_n_epochs = eval_every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *_):
        if trainer.current_epoch % self._eval_every_n_epochs == 0:
            pl_module.eval()
            losses = []
            seen = 0
            with torch.no_grad():
                for batch in self._loader:
                    inputs, targets = batch["input_ids"], batch["labels"]
                    logits = pl_module(inputs)
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)
                    loss = self._eval_fn(logits, targets)
                    losses.append(loss.detach())

                    if self._sample_size is not None:
                        seen += inputs.size(0)  # batch size
                        if seen >= self._sample_size:
                            break

            avg_loss = torch.stack(losses).mean()
            pl_module.log(f"{self._loader_name}/{self._loss_name}", avg_loss, prog_bar=False, on_step=False, on_epoch=True)
            pl_module.train()
