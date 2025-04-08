import json
from math import ceil
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset, DataLoader
from more_itertools import chunked
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor
from tokenizers import Tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(to_absolute_path(path))
    return tokenizer


class TokenizedDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self._examples = examples

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self._examples[idx].items()}

    def __len__(self):
        return len(self._examples)


class DatasetPipeline:
    def __init__(self, cfg: DictConfig, tokenizer):
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._block_size = cfg.train.model.block_size
        self._separated = cfg.data.separated_subcalls
        self._tokenize_batch_size = cfg.data.tokenize_batch_size
        self._num_workers = cfg.data.num_workers

    def _load_raw(self) -> dict[str, list[dict]]:
        result = {}
        for split, path in self._cfg.data.files.items():
            abs_path = to_absolute_path(path)
            with open(abs_path, "r") as f:
                result[split] = json.load(f)
        return result

    def _stringify_traces(self, datapoints: list[dict]) -> list[str]:
        if self._separated:
            return [
                trace
                for dp in datapoints
                for trace in [dp['solve_trace']] + dp['unit_prop_traces'] + dp['analyze_conflict_traces']
            ]
        else:
            return [dp['solve_trace_with_subcalls'] for dp in datapoints]

    @staticmethod
    def _apply_loss_mask(ids: list[int], tokenizer: Tokenizer, structural_tokens: set[str] = None,
                         block_markers: tuple[str, str] = ("READ_BEGIN", "READ_END")) -> list[int]:
        structural_ids = {tokenizer.token_to_id(tok) for tok in structural_tokens}
        begin_id = tokenizer.token_to_id(block_markers[0])
        end_id = tokenizer.token_to_id(block_markers[1])

        labels = ids[:]
        inside_block = False

        for i, tid in enumerate(ids):
            # Mask structural tokens
            if tid in structural_ids:
                labels[i] = -100
            # Mask inside block
            if tid == begin_id:
                inside_block = True
            if inside_block:
                labels[i] = -100
            if tid == end_id:
                inside_block = False

        return labels

    @staticmethod
    def _shift_and_mask_labels(ids: list[int], tokenizer: Tokenizer, structural_tokens: set[str],
                               block_markers: tuple[str, str]) -> list[int]:
        masked = DatasetPipeline._apply_loss_mask(ids, tokenizer, structural_tokens, block_markers)
        shifted = masked[1:] + [-100]  # Shift left, last token is ignored
        return shifted

    @staticmethod
    def _tokenize_texts(texts: list[str], tokenizer: Tokenizer, structural_tokens: set[str],
                        block_markers: tuple[str, str]) -> list[dict]:
        encoded = tokenizer.encode_batch(texts)
        return [
            {
                "input_ids": e.ids,
                "attention_mask": e.attention_mask,
                "labels": DatasetPipeline._shift_and_mask_labels(
                    e.ids, tokenizer, structural_tokens, block_markers
                ),
            }
            for e in encoded
        ]

    def _tokenize_batched(self, texts: list[str]) -> list[dict]:
        batches = list(chunked(texts, self._tokenize_batch_size))
        num_batches = len(batches)
        print(f"Tokenizing {len(texts)} texts using futures ({num_batches} batches)...")

        spec_toks = self._cfg.data.special_tokens
        structural_tokens = set([spec_toks['solve_markers'][0]] + 
                                [spec_toks['unit_prop_markers'][0]] +
                                [spec_toks['analyze_conflict_markers'][0]])
        block_markers = tuple(spec_toks.read_block_markers)

        with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
            futures = [
                executor.submit(
                    DatasetPipeline._tokenize_texts,
                    batch,
                    self._tokenizer,
                    structural_tokens,
                    block_markers,
                )
                for batch in batches
            ]

            tokenized = []
            for future in tqdm(futures, desc="Tokenizing"):
                tokenized.extend(future.result())

        return tokenized

    def build(self) -> dict[str, TokenizedDataset]:
        raw_data = self._load_raw()
        tokenized_splits = {}

        for split, raw_split_data in raw_data.items():
            traces = self._stringify_traces(raw_split_data)
            tokenized = self._tokenize_batched(traces)
            tokenized_splits[split] = TokenizedDataset(tokenized)

        return tokenized_splits

    def _collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self._tokenizer.token_to_id("[PAD]")

        # Filter out items that are too long
        batch = [
            item for item in batch
            if item["input_ids"].shape[0] <= self._block_size
        ]

        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def build_dataloaders(self, datasets: dict[str, TokenizedDataset]) -> dict[str, DataLoader]:
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
