import torch
from tokenizers import Tokenizer
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core.module import LightningModule
import wandb

from src.cdcl.env import AutoregressiveCDCLEnvironment
from src.dataset.dataset import TokenizedDataset
from src.model.registry import CommandRegistry
from src.model.parser import CommandParser
from src.cdcl.scratchpad import CDCLScratchpad
from src.model.inference import InferenceRunner


class InferenceCallback(Callback):
    def __init__(self, dataset: TokenizedDataset, dataset_name: str, registry: CommandRegistry, tokenizer: Tokenizer,
                 max_steps: int, sample_size: int, resample_each_time: bool, eval_every_n_epochs: int = 1):
        """
        Args:
            dataset: The dataset to sample from.
            dataset_name: Used for logging outputs.
            max_steps: Max generation steps for inference.
            sample_size: Number of examples to evaluate per epoch.
            resample_each_time: If True, resample each validation epoch; otherwise, fix samples once.
            eval_every_n_epochs: Frequency of evaluation.
        """
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._registry = registry
        self._tokenizer = tokenizer
        self._max_steps = max_steps
        self._sample_size = sample_size
        self._resample_each_time = resample_each_time
        self._eval_every_n_epochs = eval_every_n_epochs
        self._fixed_samples = None

        if not resample_each_time:
            self._fixed_samples = self._dataset.sample_full_traces(self._sample_size)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *_):
        if trainer.current_epoch % self._eval_every_n_epochs != 0:
            return

        samples = (
            self._dataset.sample_full_traces(self._sample_size)
            if self._resample_each_time or self._fixed_samples is None
            else self._fixed_samples
        )

        pl_module.eval()
        runner = InferenceRunner(pl_module)

        for i, sample in enumerate(samples):
            label_tokens = sample.solve["input_ids"]
            input_clauses_tokens = sample.input_clauses["input_ids"]

            scratchpad = CDCLScratchpad(input_clauses_tokens, self._registry)
            command_parser = CommandParser(self._registry)
            env = AutoregressiveCDCLEnvironment(self._registry, scratchpad, command_parser)

            with torch.no_grad():
                generated_ids = runner.run(env, self._max_steps)

            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=False)
            label_text = self._tokenizer.decode(label_tokens, skip_special_tokens=False)

            trainer.logger.experiment.log({
                f"inference/{self._dataset_name}/example_{i}": wandb.Html(
                    f"<b>Generated:</b><br><p>{generated_text}</p>"
                    f"<br><b>Label:</b><br><p>{label_text}</p>"
                )
            }, step=trainer.global_step)

        pl_module.train()
