from tokenizers import Tokenizer
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core.module import LightningModule
from typing import Optional
import wandb

from src.cdcl.env import AutoregressiveCDCLEnvironment
from src.dataset.dataset import CDCLDataset
from src.model.registry import CommandRegistry
from src.model.parser import CommandParser
from src.cdcl.scratchpad import CDCLScratchpad
from src.model.inference import InferenceRunner


class InferenceCallback(Callback):
    def __init__(self, dataset: CDCLDataset, dataset_name: str, registry: CommandRegistry, tokenizer: Tokenizer,
                 max_steps: int, sample_count: int, resample_each_time: bool):
        """
        Args:
            dataset: The dataset to sample from.
            dataset_name: Used for logging metrics.
            max_steps: Max generation steps for inference.
            sample_count: Number of examples to evaluate per trace type.
            resamble_each_time: If True, resample each validation epoch; otherwise, fix samples once.
        """
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._registry = registry
        self._tokenizer = tokenizer
        self._max_steps = max_steps
        self._sample_count = sample_count
        self._fixed_samples = None
        self._resample_each_time = resample_each_time

        if not resample_each_time:
            self._fixed_samples = self._dataset.sample_solve_traces(self._sample_count)
                
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._fixed_samples is None:
            samples = self._dataset.sample_solve_traces(self._sample_count)
        else:
            samples = self._fixed_samples

        correct = 0
        pl_module.eval()
        runner = InferenceRunner(pl_module)

        for sample in samples:

            input_clauses_tokens = sample.input_clauses["input_ids"]
            label_tokens = sample.compose(
                self._registry.action_cmd_tokens["CALL_UNIT_PROPAGATION"],
                self._registry.action_cmd_tokens["CALL_ANALYZE_CONFLICT"]
            )

            # setup environment
            scratchpad = CDCLScratchpad(input_clauses_tokens, self._tokenizer, self._registry)
            command_parser = CommandParser(self._registry)
            env = AutoregressiveCDCLEnvironment(self._registry, scratchpad, command_parser)

            generated_ids = runner.run(env, self._max_steps, label_tokens)

            if self._is_correct(generated_ids, label_tokens):
                correct += 1

        total = len(samples)
        acc = correct / total if total > 0 else 0.0

        last_generated_str = self._tokenizer.decode(generated_ids, skip_special_tokens=False)
        last_label_str = self._tokenizer.decode(label_tokens, skip_special_tokens=False)
        trainer.logger.log_metrics({f"{self._dataset_name}_accuracy": acc}, step=trainer.global_step)
        trainer.logger.experiment.log({
            f"inference/{self._dataset_name}/example": wandb.Html(
                f"<b>Generated:</b><br><p>{last_generated_str}</p>"
                f"<br><b>Label:</b><br><p>{last_label_str}</p>"
            )
        }, step=trainer.global_step)

    def _is_correct(self, generated: list[int], expected: list[int]) -> bool:
        """
        A trace is considered correct if:
        - The generated sequence contains exactly one occurrence of either the SAT or UNSAT token.
        - The second-to-last token in the sequence matches the expected one.
        """
        sat_id = self._registry.sat_token
        unsat_id = self._registry.unsat_token

        sat_count = generated.count(sat_id)
        unsat_count = generated.count(unsat_id)

        if sat_count + unsat_count != 1:
            return False

        if len(generated) < 2 or len(expected) < 2:
            return False

        return generated[-2] == expected[-2]