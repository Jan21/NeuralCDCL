import json
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from more_itertools import chunked
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor
from tokenizers import Tokenizer
from .dataset import CDCLDataset, TokenizedTraceExample
from src.model.registry import CommandRegistry
from dataclasses import dataclass


@dataclass
class TraceExample:
    input_clauses: str
    solve: str
    unit_propagation: list[str]
    analyze_conflict: list[str]


class DatasetPipeline:
    def __init__(self, cfg: DictConfig, tokenizer: Tokenizer, registry: CommandRegistry):
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._registry = registry
        self._block_size = cfg.train.model.block_size
        self._tokenize_batch_size = cfg.data.tokenize_batch_size
        self._num_workers = cfg.data.num_workers

    def _load_raw(self) -> dict[str, list[dict]]:
        result = {}
        for split, path in self._cfg.data.files.items():
            abs_path = to_absolute_path(path)
            with open(abs_path, "r") as f:
                result[split] = json.load(f)
        return result

    def _preprocess(self, datapoints: list[dict]) -> list[TraceExample]:
        return [ 
            TraceExample(
                input_clauses=dp["input_clauses"],
                solve=dp["solve_trace"],
                unit_propagation=dp["unit_prop_traces"],
                analyze_conflict=dp["analyze_conflict_traces"]
            )
            for dp in datapoints
        ]

    @staticmethod
    def _apply_loss_mask(ids: list[int], structural_tokens: list[int], block_markers: list[tuple[int, int]]) -> list[int]:
        labels = ids[:]
        inside_block = [False] * len(block_markers)

        for i, tid in enumerate(ids):
            # Mask structural tokens
            if tid in structural_tokens:
                labels[i] = -100

            # Check for entry/exit of each block marker
            for j, (begin_id, end_id) in enumerate(block_markers):
                if tid == begin_id:
                    inside_block[j] = True
                if inside_block[j]:
                    labels[i] = -100
                if tid == end_id:
                    inside_block[j] = False

        return labels

    @staticmethod
    def _shift_and_mask_labels(ids: list[int], structural_tokens: list[int], block_markers: list[tuple[int, int]]) -> list[int]:
        masked = DatasetPipeline._apply_loss_mask(ids, structural_tokens, block_markers)
        shifted = masked[1:] + [-100]  # Shift left, last token is ignored
        return shifted

    @staticmethod
    def _tokenize_trace_examples(
        trace_examples: list[TraceExample],
        tokenizer: Tokenizer,
        structural_tokens: list[int],
        block_markers: list[tuple[int, int]],
    ) -> list[TokenizedTraceExample]:
        
        def tokenize(text: str) -> dict:
            e = tokenizer.encode(text)
            return {
                "input_ids": e.ids,
                "attention_mask": e.attention_mask,
                "labels": DatasetPipeline._shift_and_mask_labels(
                    e.ids, structural_tokens, block_markers
                ),
            }

        def tokenize_list(texts: list[str]) -> list[dict]:
            return [tokenize(t) for t in texts]

        return [
            TokenizedTraceExample(
                input_clauses=tokenize(trace.input_clauses),
                solve=tokenize(trace.solve),
                unit_propagation=tokenize_list(trace.unit_propagation),
                analyze_conflict=tokenize_list(trace.analyze_conflict),
            )
            for trace in trace_examples
        ]

    def _tokenize_batched(self, trace_examples: list[TraceExample]) -> list[TokenizedTraceExample]:
        batches = list(chunked(trace_examples, self._tokenize_batch_size))
        num_batches = len(batches)
        print(f"Tokenizing {len(trace_examples)} trace examples using futures ({num_batches} batches)...")

        structural_tokens = [
            self._registry.solve_block_markers[0] + self._registry.up_block_markers[0] + self._registry.ac_block_markers[0]
        ]
        block_markers = [tuple(self._registry.read_block_markers)]

        with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
            futures = {
                i: executor.submit(
                    DatasetPipeline._tokenize_trace_examples,
                    batch,
                    self._tokenizer,
                    structural_tokens,
                    block_markers,
                )
                for i, batch in enumerate(batches)
            }

            tokenized = []
            for i in tqdm(range(len(futures)), desc="Tokenizing"):
                tokenized.extend(futures[i].result())

        return tokenized

    def build(self, filter_by_len: bool = True) -> dict[str, CDCLDataset]:
        raw_data = self._load_raw()
        tokenized_splits = {}

        for split, raw_split_data in raw_data.items():
            trace_examples = self._preprocess(raw_split_data)
            tokenized_examples = self._tokenize_batched(trace_examples)
            if filter_by_len:
                tokenized_examples = [
                    ex for ex in tokenized_examples
                    if all(
                        len(x["input_ids"]) <= self._block_size
                        for x in [ex.solve] + ex.unit_propagation + ex.analyze_conflict
                    )
                ]
            tokenized_splits[split] = CDCLDataset(tokenized_examples)

        return tokenized_splits

    def _collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self._registry.pad_token)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def build_dataloaders(self, datasets: dict[str, CDCLDataset]) -> dict[str, DataLoader]:
        return {
            split: DataLoader(
                dataset,
                batch_size=self._cfg.train.trainer.batch_size,
                shuffle=(split == "train"),
                num_workers=self._num_workers,
                collate_fn=self._collate_fn
            )
            for split, dataset in datasets.items()
        }
