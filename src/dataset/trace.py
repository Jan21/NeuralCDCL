from dataclasses import dataclass


@dataclass
class TraceRaw:
    input_clauses: str
    solve: str
    unit_propagation: list[str]
    analyze_conflict: list[str]


@dataclass
class TraceTokenized:
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
