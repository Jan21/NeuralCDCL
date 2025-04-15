import torch
from tokenizers import Tokenizer
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core.module import LightningModule
import wandb

from src.cdcl.env import AutoregressiveCDCLEnvironment
from src.dataset.dataset import CDCLDataset
from src.model.registry import CommandRegistry
from src.model.parser import CommandParser
from src.cdcl.scratchpad import CDCLScratchpad
from src.model.inference import InferenceRunner


class InferenceCallback(Callback):
    def __init__(self, dataset: CDCLDataset, dataset_name: str, registry: CommandRegistry, tokenizer: Tokenizer,
                 max_steps: int, sample_size: int, resample_each_time: bool, eval_every_n_epochs: int = 1):
        """
        Args:
            dataset: The dataset to sample from.
            dataset_name: Used for logging metrics.
            max_steps: Max generation steps for inference.
            sample_size: Number of examples to evaluate per trace type.
            resample_each_time: If True, resample each validation epoch; otherwise, fix samples once.
            eval_every_n_epochs: Frequency of evaluation.
        """
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._registry = registry
        self._tokenizer = tokenizer
        self._max_steps = max_steps
        self._sample_size = sample_size
        self._fixed_samples = None
        self._resample_each_time = resample_each_time
        self._eval_every_n_epochs = eval_every_n_epochs

        if not resample_each_time:
            self._fixed_samples = self._dataset.sample_solve_traces(self._sample_size)
                
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *_):
        if trainer.current_epoch % self._eval_every_n_epochs == 0:
            if self._fixed_samples is None:
                samples = self._dataset.sample_solve_traces(self._sample_size)
            else:
                samples = self._fixed_samples

            correct = 0
            pl_module.eval()
            runner = InferenceRunner(pl_module)

            for sample in samples:
                label_tokens = sample.solve["input_ids"]
                input_clauses_tokens = sample.input_clauses["input_ids"]

                # setup environment
                scratchpad = CDCLScratchpad(input_clauses_tokens, self._registry)
                command_parser = CommandParser(self._registry)
                env = AutoregressiveCDCLEnvironment(self._registry, scratchpad, command_parser)

                with torch.no_grad():
                    generated_ids = runner.run(env, self._max_steps)
                
                if self._is_correct(generated_ids, label_tokens):
                    correct += 1

            total = len(samples)
            acc = correct / total if total > 0 else 0.0
            pl_module.log(f"inference/{self._dataset_name}/accuracy", acc, prog_bar=False, on_step=False, on_epoch=True)

            last_generated_str = self._tokenizer.decode(generated_ids, skip_special_tokens=False)
            last_label_str = self._tokenizer.decode(label_tokens, skip_special_tokens=False)
            trainer.logger.experiment.log({
                f"inference/{self._dataset_name}/example": wandb.Html(
                    f"<b>Generated:</b><br><p>{last_generated_str}</p>"
                    f"<br><b>Label:</b><br><p>{last_label_str}</p>"
                )
            })
            pl_module.train()

    def _is_correct(self, generated: list[int], expected: list[int]) -> bool:
        """
        A trace is considered correct if:
        - The second-to-last token in the sequence matches the expected one.
        """
        return generated[-2] == expected[-2]