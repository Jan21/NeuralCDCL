import torch
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from typing import Optional

from src.model.registry import CommandRegistry


class EvalCallback(Callback):
    def __init__(
        self,
        loader: DataLoader,
        loader_name: str,
        registry: CommandRegistry,
        eval_every_n_epochs: int = 1,
        sample_size: Optional[int] = None,
    ):
        self._loader = loader
        self._loader_name = loader_name
        self._registry = registry
        self._eval_every_n_epochs = eval_every_n_epochs
        self._sample_size = sample_size
        self._criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *_):
        if trainer.current_epoch % self._eval_every_n_epochs != 0:
            return

        pl_module.eval()

        metrics = {
            "solve": {"loss": [], "token_correct": 0, "token_total": 0, "full_match": 0, "count": 0, "result_match": 0},
            "up": {"loss": [], "token_correct": 0, "token_total": 0, "full_match": 0, "count": 0},
            "ac": {"loss": [], "token_correct": 0, "token_total": 0, "full_match": 0, "count": 0},
        }

        seen = 0
        with torch.no_grad():
            for batch in self._loader:
                batch = {k: v.to(pl_module.device) for k, v in batch.items()}
                inputs, labels = batch["input_ids"], batch["labels"]
                logits = pl_module(inputs)  # [B, T, V]
                B, T, V = logits.shape
                logits_flat = logits.view(-1, V)
                labels_flat = labels.view(-1)

                losses = self._criterion(logits_flat, labels_flat).view(B, T)

                for i in range(B):
                    trace_type = self._get_trace_type(inputs[i][0].item())
                    if trace_type is None:
                        continue

                    self._update_metrics(metrics[trace_type], logits[i], labels[i], losses[i])

                    seen += 1
                    if self._sample_size is not None and seen >= self._sample_size:
                        break
                if self._sample_size is not None and seen >= self._sample_size:
                    break

        # Logging
        log_data = {}
        for trace_type, data in metrics.items():
            if data["count"] == 0:
                continue

            avg_loss = torch.stack(data["loss"]).mean()
            token_acc = data["token_correct"] / data["token_total"]
            full_seq_acc = data["full_match"] / data["count"]

            prefix = f"{self._loader_name}/{trace_type}"
            log_data[f"{prefix}/loss"] = avg_loss
            log_data[f"{prefix}/token_accuracy"] = token_acc
            log_data[f"{prefix}/full_sequence_accuracy"] = full_seq_acc

            if trace_type == "solve":
                result_match_acc = data["result_match"] / data["count"]
                log_data[f"{prefix}/result_match_accuracy"] = result_match_acc

        pl_module.log_dict(log_data, on_epoch=True)
        pl_module.train()

    def _get_trace_type(self, first_token: int) -> Optional[str]:
        if first_token == self._registry.tokens['structural']['solve'][0]:
            return "solve"
        elif first_token == self._registry.tokens['structural']['up'][0]:
            return "up"
        elif first_token == self._registry.tokens['structural']['ac'][0]:
            return "ac"
        else:
            return None

    def _update_metrics(self, metric: dict, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """
        Args:
            logits: Tensor of shape [T, V]
            labels: Tensor of shape [T]
            loss: Tensor of shape [T]
        """
        valid_mask = labels != -100
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        metric["loss"].append(loss[valid_mask].mean())

        preds = torch.argmax(valid_logits, dim=-1)
        correct_tokens = (preds == valid_labels).sum().item()
        total_tokens = valid_labels.size(0)

        metric["token_correct"] += correct_tokens
        metric["token_total"] += total_tokens
        metric["count"] += 1

        if total_tokens > 0 and correct_tokens == total_tokens:
            metric["full_match"] += 1

        if "result_match" in metric and len(preds) >= 2 and len(valid_labels) >= 2:
            if preds[-2].item() == valid_labels[-2].item():
                metric["result_match"] += 1
