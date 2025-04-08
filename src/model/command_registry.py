from tokenizers import Tokenizer
from omegaconf import DictConfig


class CommandRegistry:
    def __init__(self, cfg: DictConfig, tokenizer: Tokenizer):
        spec = cfg.data.special_tokens

        # Command token mappings
        self.read_cmd_tokens = {tok: tokenizer.token_to_id(tok) for tok in spec.read_cmd_tokens}
        self.write_cmd_tokens = {tok: tokenizer.token_to_id(tok) for tok in spec.write_cmd_tokens}
        self.action_cmd_tokens = {tok: tokenizer.token_to_id(tok) for tok in spec.action_cmd_tokens}

        # Block markers
        self.read_block_markers = tuple(tokenizer.token_to_id(tok) for tok in spec.read_block_markers)
        self.write_block_markers = tuple(tokenizer.token_to_id(tok) for tok in spec.write_block_markers)
        self.solve_block_markers = tuple(tokenizer.token_to_id(tok) for tok in spec.solve_markers)
        self.up_block_markers = tuple(tokenizer.token_to_id(tok) for tok in spec.unit_prop_markers)
        self.ac_block_markers = tuple(tokenizer.token_to_id(tok) for tok in spec.analyze_conflict_markers)
