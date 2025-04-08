import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.command_registry import CommandRegistry
from typing import Optional
from tokenizers import Tokenizer


class CDCLScratchpad:
    def __init__(self, input_clauses: str, tokenizer: Tokenizer, registry: CommandRegistry):
        self._saved_tokens_dict = {}
        zero_token = tokenizer.encode("0")
        self._saved_tokens_dict['level'] = zero_token

        in_clauses_tokens = tokenizer.encode(input_clauses)
        self._saved_tokens_dict['clauses'] = in_clauses_tokens

        self._read_begin_token = registry.read_block_markers[0]
        self._read_end_token = registry.read_block_markers[1]

        self._write_handlers = {
            "assignments": self._write_append,
            "reason_clauses": self._write_append,
            "decision_levels": self._write_append,
            "learned_clauses": self._write_append,
            "backtrack_level": self._write_overwrite,
            "conflict_clause": self._write_overwrite,
        }

        self._tokenizer = tokenizer

    def apply(self, cmd: str, payload: list[int]) -> Optional[list[int]]:
        if cmd.startswith("READ"):
            key = cmd.split('_')[1].lower()
            return self._handle_read(key)
        elif cmd.startswith("WRITE"):
            key = cmd.split('_')[1].lower()
            fn = self._write_handlers.get(key)
            if fn is None:
                raise ValueError(f"Unknown WRITE command: {key}")
            fn(key, payload)
        elif cmd == "BACKTRACK":
            self._backtrack()

    def _handle_read(self, key: str) -> list[int]:
        return [self._read_begin_token] + self._saved_tokens_dict.get(key, []) + [self._read_end_token]

    def _write_append(self, key: str, payload: list[int]):
        self._saved_tokens_dict[key] = payload

    def _write_overwrite(self, key: str, payload: list[int]):
        self._saved_tokens_dict[key].extend(payload)

    def render(self, key: str) -> str:
        return self._tokenizer.decode(self._saved_tokens_dict[key])

    def render_all(self) -> str:
        result = ""
        for key in self._saved_tokens_dict.keys():
            result += f"{key.upper()}: {self.render(key)}\n"
        return result

    def _backtrack(self):
        backtrack_tokens = self._saved_tokens_dict.get("backtrack_level", [])

        # Token-to-int: assuming backtrack_level is a single token (like "3")
        backtrack_level_str = self._tokenizer.decode(backtrack_tokens).strip()
        backtrack_level = int(backtrack_level_str)

        levels = self._saved_tokens_dict.get("decision_levels", [])
        assignments = self._saved_tokens_dict.get("assignments", [])
        reasons = self._saved_tokens_dict.get("reason_clauses", [])

        # Decode levels ONLY to compare as ints (since levels are small ints, not expensive)
        decoded_levels = [int(self._tokenizer.decode([t])) for t in levels]

        # Find the first index where level > backtrack_level (from the end)
        cutoff = len(decoded_levels)
        for i in reversed(range(len(decoded_levels))):
            if decoded_levels[i] > backtrack_level:
                cutoff = i
            else:
                break

        # Truncate all aligned sequences
        self._saved_tokens_dict["decision_levels"] = levels[:cutoff]
        self._saved_tokens_dict["assignments"] = assignments[:cutoff]
        self._saved_tokens_dict["reason_clauses"] = reasons[:cutoff]

        self._saved_tokens_dict["level"] = self._saved_tokens_dict["backtrack_level"]
        self._saved_tokens_dict.pop("backtrack_level")
        self._saved_tokens_dict.pop("conflict_clause")


### ================================== TEST ==================================
if __name__ == "__main__":
    from tokenizers import Tokenizer
    from omegaconf import OmegaConf
    from src.model.command_registry import CommandRegistry

    # Dummy tokenizer with vocab like { '1': 1, '2': 2, ..., 'x1': 101, '[x1 x2]': 201, ...}
    class DummyTokenizer:
        def __init__(self):
            base = {str(i): i for i in range(10)}
            vars = {f"x{i}": 100 + i for i in range(10)}
            special = {
                "READ_BEGIN": 500,
                "READ_END": 501,
                "WRITE_BEGIN": 502,
                "WRITE_END": 503,
                "WRITE_LIT": 510,
                "WRITE_REASON": 511,
                "WRITE_DECISION_LEVEL": 512,
                "WRITE_ASSIGNMENTS": 513,
                "WRITE_REASON_CLAUSES": 514,
                "WRITE_CONFLICT_CLAUSE": 515,
                "WRITE_BACKTRACK_LEVEL": 516,
                "READ_ASSIGNMENTS": 520,
                "READ_CLAUSES": 521,
                "READ_DECISION_LEVELS": 522,
                "READ_REASON_CLAUSES": 523,
                "READ_CONFLICT_CLAUSE": 524,
                "READ_LEVEL": 525,
                "BACKTRACK": 530
            }

            self.vocab = {**base, **vars, **special}
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
            }
        }
    })

    registry = CommandRegistry(cfg, tokenizer)
    context = CDCLScratchpad(input_clauses="x1 x2", tokenizer=tokenizer, registry=registry)

    # Simulate state before backtrack
    context._saved_tokens_dict["decision_levels"] = tokenizer.encode("1 1 2 3 3 7")
    context._saved_tokens_dict["assignments"] = tokenizer.encode("x1 x2 x3 x4 x5 x6")
    context._saved_tokens_dict["reason_clauses"] = tokenizer.encode("x0 x1 x2 x3 x4 x5")
    context._saved_tokens_dict["backtrack_level"] = tokenizer.encode("2")
    context._saved_tokens_dict["conflict_clause"] = tokenizer.encode("x3 x4")

    print("Before backtrack:\n" + context.render_all())

    context._backtrack()

    print("After backtrack:\n" + context.render_all())
