from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class Scratchpad:
    assignments: dict[int, bool]
    decision_levels: dict[int, int]
    reason_clauses: dict[int, list[int]]
    clauses: list[list[int]]
    learned_clauses: list[list[int]]
    level: int
    backtrack_level: Optional[int]
    conflict_clause: Optional[list[int]]

    def __init__(self):
        self.assignments = {}
        self.decision_level = {}
        self.reason_clauses = {}
        self.clauses = []
        self.learned_clauses = []
        self.level = 0
        self.backtrack_level = None
        self.conflict_clause = None

    def apply(self, command: str) -> Optional[str]:
        if command.startswith("READ"):
            if command.startswith("READ CLAUSES"):
                return self.scratchpad.clauses

            elif command.startswith("READ LEARNED_CLAUSES"):
                return self.scratchpad.learned_clauses

            elif command.startswith("READ ASSIGNMENTS"):
                return self.scratchpad.assignments

            elif command.startswith("READ DECISION_LEVELS"):
                return self.scratchpad.decision_levels

            elif command.startswith("READ REASON_CLAUSES"):
                return self.scratchpad.reason_clauses

            elif command.startswith("READ CONFLICT_CLAUSE"):
                return self.scratchpad.conflict_clause

            elif command.startswith("READ LEVEL"):
                return self.scratchpad.level

        elif command.startswith("WRITE"):
            if command.startswith("WRITE NEW_CLAUSE"):
                clause = self._parse_clause(command)
                self.scratchpad.learned_clauses.append(clause)

            if command.startswith("WRITE CONFLICT_CLAUSE"):
                clause = self._parse_clause(command)
                self.scratchpad.conflict_clause = clause

            elif command.startswith("WRITE BACKTRACK_LEVEL"):
                level = int(command.split()[-1])
                self.backtrack_level = level

            elif command.startswith("WRITE LIT"):
                # New format: "WRITE LIT x4 REASON [x1 x2 -x3]" 
                # or
                # New format: "WRITE LIT x4 REASON None" 
                lit_match = re.search(r"WRITE LIT (-?x\d+)", command)
                reason_match = re.search(r"REASON (\[.*\]|None)", command)

                if lit_match and reason_match:
                    lit_str = lit_match.group(1)
                    reason_str = reason_match.group(1)

                    lit = self._parse_literal(lit_str)
                    var, val = abs(lit), lit > 0

                    if reason_str == "None":
                        reason_clause = None
                    else:
                        reason_clause = self._parse_clause(reason_str)

                    self._assign(var, val, reason_clause)

        elif command.startswith("BACKTRACK"):
            self._backtrack()


    def _assign(self, var: int, val: bool, reason: Optional[list[int]] = None):
        self.assignments[var] = val
        self.decision_levels[var] = self.level
        if reason:
            self.reason_clauses[var] = reason

    def _parse_clause(self, line: str) -> list[int]:
        return [int(x.replace("x", "").replace("-", "-")) for x in re.findall(r"-?x\d+", line)]

    def _parse_literal(self, lit_str: str) -> int:
        lit = lit_str.strip().replace("x", "")
        return int(lit)

    def _backtrack(self):
        self.level = self.backtrack_level
        self.assignments = {var: value for var, value in self.assignments.items() 
                          if self.decision_level[var] <= self.level}
        self.reason_clauses = {var: clause for var, clause in self.reason_clauses.items() 
                             if var in self.assignments}
        self.backtrack_level = None
        self.conflict_clause = None