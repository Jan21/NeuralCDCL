from src.model.registry import CommandRegistry
from typing import Optional


class CommandParser:
    def __init__(self, registry: CommandRegistry):
        self.reg = registry
        self.state = "idle"
        self.command = None # e.g., "READ_ASSIGNMENTS" or "WRITE_LIT"
        self.payload = [] # tokens between WRITE_BEGIN/WRITE_END

        self.write_begin_token = registry.tokens['structural']['write'][0]
        self.write_end_token = registry.tokens['structural']['write'][1]

        # Flatten all commands into one mapping: token_id -> command_name
        self.token_to_cmd = {
            **{v: k for k, v in registry.tokens['commands']['read'].items()},
            **{v: k for k, v in registry.tokens['commands']['write'].items()},
            **{v: k for k, v in registry.tokens['commands']['action'].items()},
        }

    def reset(self):
        self.state = "idle"
        self.command = None
        self.payload.clear()

    def step(self, token: int) -> Optional[tuple[str, list[int]]]:
        # === STATE 1: Waiting for command ===
        if self.state == "idle":
            if token in self.token_to_cmd:
                name = self.token_to_cmd[token]
                if name.startswith("WRITE"):
                    self.command = name
                    self.state = "reading"
                    return None
                else:
                    return (name, [])

        # === STATE 2: Waiting for BEGIN ===
        elif self.state == "reading":
            if token == self.write_begin_token:
                self.state = "payload"

        # === STATE 3: Capturing payload ===
        elif self.state == "payload":
            if token == self.write_end_token:
                result = (self.command, self.payload.copy())
                self.reset()
                return result
            else:
                self.payload.append(token)

        return None