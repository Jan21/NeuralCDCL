import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.registry import CommandRegistry
from typing import Optional
from tokenizers import Tokenizer


class CDCLScratchpad:
    def __init__(self, input_clauses: str, tokenizer: Tokenizer, registry: CommandRegistry):
        self._saved_tokens_dict = {}
        self._saved_tokens_dict['level'] = []

        in_clauses_tokens = tokenizer.encode(input_clauses).ids
        self._saved_tokens_dict['clauses'] = in_clauses_tokens

        self._read_begin_token = registry.read_block_markers[0]
        self._read_end_token = registry.read_block_markers[1]

        self._semantic_begin_token = registry.semantic_block_markers[0]
        self._semantic_end_token = registry.semantic_block_markers[1]

        self._counter_unit_token = registry.counter_unit_token

        self._handlers = {
            "assignments": {"type": "list", "write": self._write_append, "wrap_tokens": False},
            "reason_clauses": {"type": "list", "write": self._write_append, "wrap_tokens": True},
            "decision_levels": {"type": "list", "write": self._write_append, "wrap_tokens": True},
            "learned_clauses": {"type": "list", "write": self._write_append, "wrap_tokens": True},
            "backtrack_level": {"type": "element", "write": self._write_overwrite},
            "conflict_clause": {"type": "element", "write": self._write_overwrite},
            "level": {"type": "element", "write": None}, # read-only, only LEVEL_UP action can modify
            "clauses": {"type": "element", "write": None}, # read-only, cannot be modified
        }

        self._tokenizer = tokenizer

    def apply(self, cmd: str, payload: list[int]) -> Optional[list[int]]:
        if cmd.startswith("READ"):
            key = '_'.join(cmd.split('_')[1:]).lower()
            return self._handle_read(key)
        elif cmd.startswith("WRITE"):
            key = '_'.join(cmd.split('_')[1:]).lower()
            fn = self._handlers[key]['write']
            if fn is None:
                raise ValueError(f"Unknown WRITE command: {key}")
            fn(key, payload)
        elif cmd == "BACKTRACK":
            self._backtrack()
        elif cmd == "LEVEL_UP":
            self._saved_tokens_dict['level'].append(self._counter_unit_token)

    def _handle_read(self, key: str) -> list[int]:
        handler = self._handlers[key]
        if handler['type'] == "element":
            content = self._saved_tokens_dict.get(key, [])
        elif handler['type'] == "list":
            list_of_lists = self._saved_tokens_dict.get(key, [])
            content = [
                token
                for element in list_of_lists
                for token in (
                    [self._semantic_begin_token] + element + [self._semantic_end_token]
                    if handler["wrap_tokens"]
                    else element
                )
            ]
        else:
            raise ValueError(f"Unknown READ command: {key}")
        return [self._read_begin_token] + content + [self._read_end_token]

    def _write_append(self, key: str, payload: list[int]):
        if key not in self._saved_tokens_dict:
            self._saved_tokens_dict[key] = []
        self._saved_tokens_dict[key].append(payload)

    def _write_overwrite(self, key: str, payload: list[int]):
        self._saved_tokens_dict[key] = payload

    def render(self, key: str) -> str:
        return self._tokenizer.decode(self._handle_read(key))

    def render_all(self) -> str:
        result = ""
        for key in self._saved_tokens_dict.keys():
            result += f"{key.upper()}: {self.render(key)}\n"
        return result

    def _backtrack(self):
        if "backtrack_level" not in self._saved_tokens_dict:
            return

        backtrack_tokens = self._saved_tokens_dict["backtrack_level"]
        backtrack_level = len(backtrack_tokens)

        decision_levels = self._saved_tokens_dict.get("decision_levels", [])
        assignments = self._saved_tokens_dict.get("assignments", [])
        reasons = self._saved_tokens_dict.get("reason_clauses", [])

        decoded_decision_levels = [len(t) for t in decision_levels]

        # Find the first index where level > backtrack_level (from the end)
        cutoff = len(decoded_decision_levels)
        for i in reversed(range(len(decoded_decision_levels))):
            if decoded_decision_levels[i] > backtrack_level:
                cutoff = i
            else:
                break

        # Truncate all aligned sequences
        self._saved_tokens_dict["decision_levels"] = decision_levels[:cutoff]
        self._saved_tokens_dict["assignments"] = assignments[:cutoff]
        self._saved_tokens_dict["reason_clauses"] = reasons[:cutoff]

        self._saved_tokens_dict["level"] = self._saved_tokens_dict["backtrack_level"]
        self._saved_tokens_dict.pop("backtrack_level")
        self._saved_tokens_dict.pop("conflict_clause")


### ================================== TEST ==================================
if __name__ == "__main__":
    from tokenizers import Tokenizer
    from omegaconf import OmegaConf
    from model.registry import CommandRegistry

    # Dummy tokenizer with vocab like { '1': 1, '2': 2, ..., 'x1': 101, '[x1 x2]': 201, ...}
    class DummyTokenizer:
        def __init__(self):
            base = {str(i): i for i in range(10)}
            vars = {f"x{i}": 100 + i for i in range(10)}
            neg_vars = {f"-x{i}": 100 + i for i in range(10)}
            special = {
                "READ_BEGIN": 500,
                "READ_END": 501,
                "WRITE_BEGIN": 502,
                "WRITE_END": 503,
                "WRITE_ASSIGNMENTS_APPEND": 510,
                "WRITE_REASON_CLAUSES_APPEND": 511,
                "WRITE_DECISION_LEVELS_APPEND": 512,
                "WRITE_CONFLICT_CLAUSE": 515,
                "WRITE_BACKTRACK_LEVEL": 516,
                "READ_ASSIGNMENTS": 520,
                "READ_CLAUSES": 521,
                "READ_DECISION_LEVELS": 522,
                "READ_REASON_CLAUSES": 523,
                "READ_CONFLICT_CLAUSE": 524,
                "READ_LEVEL": 525,
                "BACKTRACK": 530,
                "[": 531,
                "]": 532,
                "None": 533,
                "I": 534
            }

            self.vocab = {**base, **vars, **neg_vars, **special}
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [self.vocab[t] for t in tokens]

        def decode(self, ids: list[int]):
            return " ".join(self.reverse_vocab.get(i, f"[UNK:{i}]") for i in ids)

        def token_to_id(self, tok: str) -> int:
            return self.vocab[tok]


    tokenizer = DummyTokenizer()

    # Fake config
    cfg = OmegaConf.create({
        "data": {
            "read_block_markers": ["READ_BEGIN", "READ_END"],
            "write_block_markers": ["WRITE_BEGIN", "WRITE_END"],
            "special_tokens": {
                "read_cmd_tokens": [],
                "write_cmd_tokens": [],
                "action_cmd_tokens": [],
                "read_block_markers": ["READ_BEGIN", "READ_END"],
                "write_block_markers": ["WRITE_BEGIN", "WRITE_END"],
                "solve_markers": [],
                "unit_prop_markers": [],
                "analyze_conflict_markers": [],
                "semantic_markers": ["[", "]"],
                "counter_unit_token": "I"
            }
        }
    })

    registry = CommandRegistry(cfg, tokenizer)
    context = CDCLScratchpad(input_clauses="[ x1 x2 x3 ] [ -x1 x2 -x3 ]", tokenizer=tokenizer, registry=registry)

    # Simulate state before backtrack
    context._saved_tokens_dict["level"] = tokenizer.encode("I I I")
    context._saved_tokens_dict["decision_levels"] = [tokenizer.encode("I"), tokenizer.encode("I I"), tokenizer.encode("I I I")]
    context._saved_tokens_dict["assignments"] = [tokenizer.encode("x1"),  tokenizer.encode("-x2"), tokenizer.encode("x3")]
    context._saved_tokens_dict["reason_clauses"] = [tokenizer.encode("None"), tokenizer.encode("None"), tokenizer.encode("x1 x2 x3")]
    context._saved_tokens_dict["backtrack_level"] = tokenizer.encode("I")
    context._saved_tokens_dict["conflict_clause"] = tokenizer.encode("-x1 x2 -x3")

    print("Before backtrack:\n" + context.render_all())

    context._backtrack()

    print("After backtrack:\n" + context.render_all())
