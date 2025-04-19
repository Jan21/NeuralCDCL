from src.model.registry import CommandRegistry
from src.cdcl.scratchpad import CDCLScratchpad
from src.model.parser import CommandParser


class AutoregressiveCDCLEnvironment:
    def __init__(self, registry: CommandRegistry, scratchpad: CDCLScratchpad, command_parser: CommandParser):
        self._scratchpad = scratchpad

        self._history = [registry.tokens['structural']['solve'][0]]
        self._command_parser = command_parser
        self._stashed_history = None

        # Fast access tokens
        self._unit_prop_begin = registry.tokens['structural']['up'][0]
        self._unit_prop_end = registry.tokens['structural']['up'][1]
        self._ac_begin = registry.tokens['structural']['ac'][0]
        self._ac_end = registry.tokens['structural']['ac'][1]
        self._solve_end = registry.tokens['structural']['solve'][1]

        self._call_up = registry.tokens['commands']['action']["CALL_UNIT_PROPAGATION"]
        self._call_ac = registry.tokens['commands']['action']["CALL_ANALYZE_CONFLICT"]

    def append(self, token: int):
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
        if token == self._call_up:
            self._stash_and_reset(self._unit_prop_begin)
        elif token == self._call_ac:
            self._stash_and_reset(self._ac_begin)
        elif token in [self._unit_prop_end, self._ac_end]:
            self._restore_history()

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

    def is_finished(self) -> bool:
        return self._history[-1] == self._solve_end
