import torch
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from typing import Callable


class EvalLossCallback(Callback):
    def __init__(
        self,
        loader: DataLoader,
        loader_name: str,
        eval_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
        eval_every_n_steps: int,
        loss_name: str
    ):
        """
        Args:
            loader: The dataloader to use for evaluation.
            loader_name: The dataloader's name.
            eval_fn: A function that takes (pl_module, batch) and returns a scalar torch.Tensor loss.
            eval_every_n_epochs: Frequency of evaluation.
            loss_name: Name for logging.
        """
        self._loader = loader
        self._loader_name = loader_name
        self._eval_fn = eval_fn
        self._eval_every_n_steps = eval_every_n_steps
        self._loss_name = loss_name

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule):
        step = trainer.global_step
        if step % self._eval_every_n_steps == 0:
            pl_module.eval()
            losses = []
            with torch.no_grad():
                for batch in self._loader:
                    loss = self._eval_fn(pl_module, batch)
                    losses.append(loss)
            avg_loss = torch.stack(losses).mean()
            print(f"[Step {step}] {self._loader_name}_{self._loss_name}: {avg_loss.item():.4f}")
            trainer.logger.log_metrics({f"{self._loader_name}_{self._loss_name}": avg_loss.item()}, step=trainer.global_step)
            pl_module.train()
