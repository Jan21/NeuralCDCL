from torch.utils.data import Dataset
from typing import Optional, Literal
import torch
import random
from dataclasses import dataclass


@dataclass
class TokenizedTraceExample:
    input_clauses: dict
    solve: dict
    unit_propagation: list[dict]
    analyze_conflict: list[dict]

    def compose(self, up_call_token: int, ac_call_token: int) -> list[int]:
        composed = []
        up_idx = 0
        ac_idx = 0

        solve_ids = self.solve["input_ids"]

        for token in solve_ids:
            composed.append(token)
            if token == up_call_token:
                composed.extend(self.unit_propagation[up_idx]["input_ids"])
                up_idx += 1
            elif token == ac_call_token:
                composed.extend(self.analyze_conflict[ac_idx]["input_ids"])
                ac_idx += 1

        return composed


class CDCLDataset(Dataset):
    def __init__(self, examples: list[TokenizedTraceExample]):
        self.examples = examples
        self._length = sum(1 + len(e.unit_propagation) + len(e.analyze_conflict) for e in examples)

        # Mapping from flat index (example_idx, trace_type, sub_idx)
        # trace_type is one of {"solve", "up", "ac"}
        self._index_map: list[tuple[int, Literal["solve", "up", "ac"], Optional[int]]] = []

        for i, ex in enumerate(examples):
            self._index_map.append((i, "solve", None))
            self._index_map.extend((i, "up", j) for j in range(len(ex.unit_propagation)))
            self._index_map.extend((i, "ac", j) for j in range(len(ex.analyze_conflict)))

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        example_idx, kind, sub_idx = self._index_map[idx]
        ex = self.examples[example_idx]

        if kind == "solve":
            trace = ex.solve
        elif kind == "up":
            trace = ex.unit_propagation[sub_idx]
        elif kind == "ac":
            trace = ex.analyze_conflict[sub_idx]
        else:
            raise ValueError(f"Unknown trace kind: {kind}")

        return {k: torch.tensor(v) for k, v in trace.items()}

    def sample_solve_traces(self, n: int) -> list[TokenizedTraceExample]:
        """
        Randomly samples up to `n` solve traces with their corresponding input_clauses and tokenized data.
        """
        solve_indices = [
            idx for idx, (ex_idx, kind, _) in enumerate(self._index_map)
            if kind == "solve"
        ]

        sampled = random.sample(solve_indices, min(n, len(solve_indices)))

        return [
            self.examples[self._index_map[idx][0]]
            for idx in sampled
        ]
