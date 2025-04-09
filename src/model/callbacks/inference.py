import torch
from tokenizers import Tokenizer
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core.module import LightningModule
from typing import Optional

from model.inference import InferenceRunner
from cdcl.env import AutoregressiveCDCLEnvironment
from dataset.dataset import CDCLDataset
from model.registry import CommandRegistry
from model.parser import CommandParser
from src.cdcl_env.cdcl_scratchpad import CDCLScratchpad


class InferenceCallback(Callback):
    def __init__(self, dataset: CDCLDataset, dataset_name: str, registry: CommandRegistry, tokenizer: Tokenizer,
                 max_steps: int, sample_count: int, resample_each_time: bool, 
                 seed: Optional[int] = None):
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
        self._seed = seed if seed else 43
        self._resample_each_time = resample_each_time

        if not resample_each_time:
            self._fixed_samples = self._dataset.sample_by_type('solve', self._sample_count, seed=self._seed)
                
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._fixed_samples is None:
            samples = self._dataset.sample_by_type('solve', self._sample_count, seed=self._seed)
        else:
            samples = self._fixed_samples

        correct = 0
        total = 0

        pl_module.eval()
        runner = InferenceRunner(pl_module)

        for trace_type in self.trace_types:
            for sample in samples[trace_type]:
                input_clauses_str, full_trace_tokenized = sample
                label = full_trace_tokenized["input_ids"].tolist()

                # setup environment
                scratchpad = CDCLScratchpad(input_clauses_str, self._tokenizer, self._registry)
                command_parser = CommandParser(self._registry)
                env = AutoregressiveCDCLEnvironment(self._registry, scratchpad, command_parser)

                generated_ids = runner.run(env, self._max_steps)

            if self._is_correct(generated_ids, label):
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        trainer.logger.log_metrics({f"{self._dataset_name}_accuracy": acc}, step=trainer.global_step)
        trainer.logger.experiment.log({
            "example/generated": self._tokenizer.decode(generated_ids),
            "example/expected": self._tokenizer.decode(label),
            "step": trainer.global_step
        })

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