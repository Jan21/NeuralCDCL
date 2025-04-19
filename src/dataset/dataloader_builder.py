import os
import torch
from torch.utils.data import DataLoader
from src.dataset.dataset import TokenizedDataset
from src.model.registry import CommandRegistry
from typing import Optional


class DataloaderBuilder:
    def __init__(
        self, 
        batch_size: int, 
        registry: CommandRegistry, 
        num_workers: Optional[int] = None, 
        device: Optional[torch.device] = None
    ):
        self._batch_size = batch_size
        self._registry = registry
        self._num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() // 2)
        self._device = device or torch.device("cpu")

    def _collate_fn(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"], device=self._device) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"], device=self._device) for item in batch]
        labels = [torch.tensor(item["labels"], device=self._device) for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self._registry.tokens['pad'])
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def build_dataloader(self, dataset: TokenizedDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=(self._device.type == "cuda")
        )
