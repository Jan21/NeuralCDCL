import re
from typing import List
from scratchpad import Scratchpad

class TraceEnvironment:
    def __init__(self):
        self.scratchpad = Scratchpad(
            assignments={}, decision_level={}, reason_clauses={},
            clauses=[], learned_clauses=[], level=0
        )

    def apply(self, command: str):
        if command.startswith("READ CLAUSES"):
            self.scratchpad.clauses = self._parse_clause_list(command)

        elif command.startswith("READ LEARNED_CLAUSES"):
            self.scratchpad.learned_clauses = self._parse_clause_list(command)

        elif command.startswith("READ ASSIGNMENTS"):
            self.scratchpad.assignments = self._parse_assignments(command)

        elif command.startswith("WRITE PICKED_LIT"):
            lit = self._parse_literal(command)
            var, val = abs(lit), lit > 0
            self._assign(var, val, reason=None)

        elif command.startswith("WRITE LEARNED_LIT"):
            lit = self._parse_literal(command)
            var, val = abs(lit), lit > 0
            self._assign(var, val, reason="unit")  # just a tag

        elif command.startswith("WRITE NEW_CLAUSE"):
            clause = self._parse_clause(command)
            self.scratchpad.learned_clauses.append(clause)

        elif command == "BACKTRACK":
            self._backtrack_one_level()

        elif command.startswith("WRITE BACKTRACK_LEVEL"):
            level = int(command.split()[-1])
            self._backtrack_to(level)

        elif command.startswith("READ LEVEL"):
            self.scratchpad.level = int(command.split()[-1])

        elif command.startswith("WRITE LEARNED_LIT"):
            # already handled above
            pass

    ### ----------------- Helpers -----------------

    def _assign(self, var: int, val: bool, reason):
        self.scratchpad.assignments[var] = val
        self.scratchpad.decision_level[var] = self.scratchpad.level
        if reason:
            self.scratchpad.reason_clauses[var] = reason

    def _backtrack_one_level(self):
        self._backtrack_to(self.scratchpad.level - 1)

    def _backtrack_to(self, level: int):
        self.scratchpad.level = level
        self.scratchpad.assignments = {
            var: val for var, val in self.scratchpad.assignments.items()
            if self.scratchpad.decision_level[var] <= level
        }
        self.scratchpad.decision_level = {
            var: dl for var, dl in self.scratchpad.decision_level.items()
            if dl <= level
        }
        self.scratchpad.reason_clauses = {
            var: reason for var, reason in self.scratchpad.reason_clauses.items()
            if var in self.scratchpad.assignments
        }

    def _parse_clause(self, line: str) -> List[int]:
        return [int(x.replace("x", "").replace("-", "-")) for x in re.findall(r"-?x\d+", line)]

    def _parse_clause_list(self, line: str) -> List[List[int]]:
        return [self._parse_clause(cl) for cl in re.findall(r"\[([^\[\]]+)\]", line)]

    def _parse_assignments(self, line: str) -> dict:
        lits = self._parse_clause(line)
        return {abs(l): l > 0 for l in lits}

    def _parse_literal(self, line: str) -> int:
        match = re.search(r"(-?x\d+)", line)
        if match:
            return int(match.group(1).replace("x", "").replace("-", "-"))
        raise ValueError(f"Invalid literal in line: {line}")
