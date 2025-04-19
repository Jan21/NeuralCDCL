from torch.utils.data import Dataset
from typing import Optional, Literal
from typing import Optional, Union, Iterable, Literal
import torch
import random

from src.dataset.trace import TraceTokenized


class TokenizedDataset(Dataset):
    def __init__(
        self, 
        examples: list[TraceTokenized], 
        kind: Optional[Union[Literal["solve", "up", "ac"], Iterable[Literal["solve", "up", "ac"]]]] = None
    ):
        self._examples = examples
        self._kind_filter = self._normalize_kind(kind)
        self._rebuild_index_map()

    def _normalize_kind(self, kind) -> set[str]:
        if kind is None:
            return {"solve", "up", "ac"}
        if isinstance(kind, str):
            return {kind}
        return set(kind)

    def _rebuild_index_map(self):
        self._index_map = []

        for i, ex in enumerate(self._examples):
            if "solve" in self._kind_filter:
                self._index_map.append((i, "solve", None))
            if "up" in self._kind_filter:
                self._index_map.extend((i, "up", j) for j in range(len(ex.unit_propagation)))
            if "ac" in self._kind_filter:
                self._index_map.extend((i, "ac", j) for j in range(len(ex.analyze_conflict)))

        self._length = len(self._index_map)

    def filter_by_block_size(self, max_len: int) -> "TokenizedDataset":
        def is_valid(ex: TraceTokenized) -> bool:
            return all(len(x["input_ids"]) <= max_len for x in [ex.solve] + ex.unit_propagation + ex.analyze_conflict)

        init_len = len(self.examples)
        self._examples = [ex for ex in self._examples if is_valid(ex)]
        self._rebuild_index_map()
        print(f"Filtered dataset to {len(self._examples)} examples (max_len = {max_len}, original_length = {init_len})")
        return self

    @property
    def examples(self) -> list[TraceTokenized]:
        return self._examples

    @property
    def kind(self) -> set[str]:
        return self._kind_filter

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx) -> dict[str, list[int]]:
        example_idx, kind, sub_idx = self._index_map[idx]
        ex = self._examples[example_idx]

        if kind == "solve":
            trace = ex.solve
        elif kind == "up":
            trace = ex.unit_propagation[sub_idx]
        elif kind == "ac":
            trace = ex.analyze_conflict[sub_idx]
        else:
            raise ValueError(f"Unknown trace kind: {kind}")

        return trace

    def sample_full_traces(self, n: int) -> list[TraceTokenized]:
        """
        Randomly samples up to `n` solve traces with their corresponding input_clauses and tokenized data.
        """
        solve_indices = [i for i, (_, kind, _) in enumerate(self._index_map) if kind == "solve"]
        sampled = random.sample(solve_indices, min(n, len(solve_indices)))
        return [self._examples[self._index_map[i][0]] for i in sampled]

    def save(self, path: str) -> None:
        torch.save({
            "examples": self._examples,
            "kind_filter": self._kind_filter
        }, path)

    @staticmethod
    def load(path: str) -> "TokenizedDataset":
        data = torch.load(path)
        return TokenizedDataset(data["examples"], kind=data["kind_filter"])

    def __repr__(self):
        return (
            f"<TokenizedDataset with {len(self._examples)} examples, "
            f"{self._length} total subtraces, kind={sorted(self._kind_filter)}>"
        )
