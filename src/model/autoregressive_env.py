from src.model.command_registry import CommandRegistry
from src.cdcl_env.cdcl_scratchpad import CDCLScratchpad
from src.model.command_parser import CommandParser


class AutoregressiveCDCLEnvironment:
    def __init__(self, registry: CommandRegistry, scratchpad: CDCLScratchpad, command_parser: CommandParser):
        self._registry = registry
        self._scratchpad = scratchpad

        self._history = []
        self._command_parser = command_parser
        self._stashed_history = None

        # Fast access tokens
        self._unit_prop_begin = self._registry.up_block_markers[0]
        self._unit_prop_end = self._registry.up_block_markers[1]
        self._ac_begin = self._registry.ac_block_markers[0]
        self._ac_end = self._registry.ac_block_markers[1]

    def append(self, token: int) -> list[int]:
        """
        Process a single token emitted by the model. Returns the current input to feed back.
        """
        # Append token to history
        self._history.append(token)

        # Try to parse a command
        parsed = self._command_parser.step(token)
        if parsed is not None:
            command, payload = parsed
            response = self._scratchpad.apply(command, payload)
            if response is not None:
                self._history.extend(response)  # Inline injection of READ response

        # Handle UNIT_PROPAGATION / ANALYZE_CONFLICT switching
        if token in [self._unit_prop_begin, self._ac_begin]:
            self._stash_and_reset(token)
        elif token in [self._unit_prop_end, self._ac_end]:
            self._restore_history()

        return self._get_current_input()

    def get_current_input(self) -> list[int]:
        return self._history

    def _stash_and_reset(self, begin_token: int):
        self._stashed_history = self._history.copy()
        self._history = [begin_token]
        self._command_parser.reset()

    def _restore_history(self):
        if self._stashed_history is not None:
            self._history = self._stashed_history
            self._stashed_history = None
            self._command_parser.reset()
