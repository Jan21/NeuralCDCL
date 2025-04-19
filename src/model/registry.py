from tokenizers import Tokenizer
from omegaconf import DictConfig


class CommandRegistry:
    def __init__(self, cfg: DictConfig, tokenizer: Tokenizer):
        tokens = cfg.tokens

        self._tokens = {
            'pad': tokenizer.token_to_id("[PAD]"),
            'counter_unit_token': tokenizer.token_to_id(tokens.labels.counter_unit_token),
            'sat_token': tokenizer.token_to_id(tokens.labels.sat_token),
            'unsat_token': tokenizer.token_to_id(tokens.labels.unsat_token),

            'structural': {
                'read': tuple(tokenizer.token_to_id(tok) for tok in tokens.structural.read_block),
                'write': tuple(tokenizer.token_to_id(tok) for tok in tokens.structural.write_block),
                'solve': tuple(tokenizer.token_to_id(tok) for tok in tokens.structural.solve_block),
                'up': tuple(tokenizer.token_to_id(tok) for tok in tokens.structural.unit_prop_block),
                'ac': tuple(tokenizer.token_to_id(tok) for tok in tokens.structural.analyze_conflict_block),
                'semantic': tuple(tokenizer.token_to_id(tok) for tok in tokens.structural.semantic_block),
            },

            'commands': {
                'read': {
                    token_str: tokenizer.token_to_id(token_str)
                    for token_str in tokens.commands.read.values()
                },
                'write': {
                    token_str: tokenizer.token_to_id(token_str)
                    for token_str in tokens.commands.write.values()
                },
                'action': {
                    token_str: tokenizer.token_to_id(token_str)
                    for token_str in tokens.commands.action.values()
                }
            },
        }

    @property
    def tokens(self) -> dict:
        return self._tokens
