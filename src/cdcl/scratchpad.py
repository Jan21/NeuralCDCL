import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.registry import CommandRegistry
from typing import Optional
from tokenizers import Tokenizer


class CDCLScratchpad:
    def __init__(self, tokenized_input_clauses: list[int], registry: CommandRegistry):
        self._saved_tokens_dict = {}
        self._saved_tokens_dict['level'] = []

        self._saved_tokens_dict['clauses'] = tokenized_input_clauses

        self._read_begin_token = registry.tokens['structural']['read'][0]
        self._read_end_token = registry.tokens['structural']['read'][1]

        self._semantic_begin_token = registry.tokens['structural']['semantic'][0]
        self._semantic_end_token = registry.tokens['structural']['semantic'][1]

        self._counter_unit_token = registry.tokens['counter_unit_token']

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

    def render(self, key: str, tokenizer: Tokenizer) -> str:
        return tokenizer.decode(self._handle_read(key))

    def render_all(self) -> str:
        result = ""
        for key in self._saved_tokens_dict.keys():
            result += f"{key.upper()}: {self.render(key)}\n"
        return result

    def _backtrack(self):
        if "backtrack_level" not in self._saved_tokens_dict or "conflict_clause" not in self._saved_tokens_dict:
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
