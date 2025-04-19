import os
from more_itertools import chunked
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from tokenizers import Tokenizer
from src.model.registry import CommandRegistry
from src.dataset.trace import TraceRaw, TraceTokenized
from src.dataset.dataset import TokenizedDataset
from src.dataset.tokenization import tokenize_trace_batch_wrapper, tokenize_trace_batch
from typing import Optional


class DatasetBuilder:
    def __init__(
        self, 
        tokenizer: Tokenizer, 
        registry: CommandRegistry,
        tokenize_batch_size: int,
        num_workers: Optional[int] = None
    ):
        self._tokenizer = tokenizer
        self._registry = registry
        self._tokenize_batch_size = tokenize_batch_size
        self._num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() // 2)

    def _preprocess(self, datapoints: list[dict]) -> list[TraceRaw]:
        return [ 
            TraceRaw(
                input_clauses=dp["input_clauses"],
                solve=dp["solve_trace"],
                unit_propagation=dp["unit_prop_traces"],
                analyze_conflict=dp["analyze_conflict_traces"]
            )
            for dp in datapoints
        ]

    def _tokenize_batched(self, trace_examples: list[TraceRaw]) -> list[TraceTokenized]:
        batches = list(chunked(trace_examples, self._tokenize_batch_size))

        struct_tokens = self._registry.tokens['structural'] 
        single_tokens = [struct_tokens['solve'][0], struct_tokens['up'][0], struct_tokens['ac'][0]]
        block_tokens = [struct_tokens['read']]

        tokenized = []
        desc = f"Tokenizing {len(trace_examples)} examples in {len(batches)} batches"

        if self._num_workers < 2:
            for batch in tqdm(batches, desc=desc):
                tokenized.extend(tokenize_trace_batch(batch, self._tokenizer, single_tokens, block_tokens))
        else:
            with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
                for tokenized_batch in tqdm(
                    executor.map(
                        tokenize_trace_batch_wrapper,
                        [(batch, self._tokenizer, single_tokens, block_tokens) for batch in batches]
                    ),
                    total=len(batches),
                    desc=desc
                ):
                    tokenized.extend(tokenized_batch)

        return tokenized

    def build(self, raw_data: list[dict]) -> TokenizedDataset:
        trace_examples = self._preprocess(raw_data)
        tokenized_examples = self._tokenize_batched(trace_examples)
        return TokenizedDataset(tokenized_examples)
