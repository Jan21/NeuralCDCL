from typing import Optional
import re

class Tracer:
    """
    A tracer that maintains separate traces for:
      1) Solve
      2) Unit Propagation
      3) Analyze Conflict

    Each submodule's logs are appended to a sub-trace. Then the solve trace can selectively
    embed only the submodule ARGUMENTS...START plus a single RESULTS line.
    """
    def __init__(self):
        # The solve trace is top-level and unique
        self.solve_trace: list[str] = []

        # We may have multiple calls to unit propagation / conflict analysis within one solve
        self.unit_propagation_traces: list[list[str]] = []
        self.analyze_conflict_traces: list[list[str]] = []

        # We track the *active* submodule trace (if any)
        self.current_unit_trace: Optional[list[str]] = None
        self.current_analyze_trace: Optional[list[str]] = None

        self.input_clauses_trace = None

    ### ANALYZE_CONFLICT TRACES ###
    def on_analyze_conflict_start(self, assignments: dict, decision_level: dict, reason_clauses: dict, 
                                  conflict_clause: list, level: int):
        self.current_analyze_trace = []
        decision_level_adj = [[decision_level[var]] for var in assignments.keys()]
        reason_clauses_adj = [reason_clauses[var] if var in reason_clauses else '[ None ]' for var in assignments.keys()]
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"ANALYZE_CONFLICT_BEGIN",
            f"READ_ASSIGNMENTS READ_BEGIN {format_list(assignments_lst, is_var=True)} READ_END",
            f"READ_DECISION_LEVELS READ_BEGIN {format_list(decision_level_adj, is_var=False, use_unary=True)} READ_END",
            f"READ_REASON_CLAUSES READ_BEGIN {format_list(reason_clauses_adj, is_var=True)} READ_END",
            f"READ_CONFLICT_CLAUSE READ_BEGIN {format_list(conflict_clause, is_var=True)} READ_END",
            f"READ_LEVEL READ_BEGIN {encode_number_unary(level)} READ_END",
        ]
        self.current_analyze_trace.extend(trace) 

    def on_analyze_conflict_iteration_end(self, queue: list, curr_level_vars: set, learned_lits: set, 
                                          is_uip: bool, selected_var: Optional[int] = None, reason_clause: Optional[list] = None):
        trace = [
            f"QUEUE {format_list(queue, is_var=True)}",
            f"RESOLVING {format_lit(selected_var) if selected_var is not None else 'None'}",
            f"REASON_CLAUSE {format_list(reason_clause, is_var=True) if reason_clause is not None else 'None'}",
            f"CURRENT_LVL_VARS {format_list(list(curr_level_vars), is_var=True)}",
            f"LEARNED_LITS {format_list(list(learned_lits), is_var=True)}",
            f"IS_UIP {str(int(is_uip))}",
        ]
        self.current_analyze_trace.extend(trace) 

    def on_analyze_conflict_end(self, new_clause: list):
        trace = [
            f"WRITE_LEARNED_CLAUSES WRITE_BEGIN {format_list(new_clause, is_var=True)} WRITE_END",
        ]
        self.current_analyze_trace.extend(trace) 

    def on_analyze_conflict_find_backtrack_level_end(self, learned_clause: list, goto_level: int, levels: Optional[list] = None):
        trace = [
            f"DECISION_LEVELS {format_list(list(map(lambda x: [x], levels)) if levels else [], is_var=False, use_unary=True)}",
            f"WRITE_BACKTRACK_LEVEL WRITE_BEGIN {encode_number_unary(goto_level)} WRITE_END",
            f"ANALYZE_CONFLICT_END"
        ]
        self.current_analyze_trace.extend(trace) 

        self.analyze_conflict_traces.append(self.current_analyze_trace)
        self.current_analyze_trace = None


    ### UNIT_PROPAGATION TRACES ###
    def on_unit_propagation_start(self, clauses: list, learned_clauses: list, assignments: dict):
        self.current_unit_trace = []
        clauses_combined = clauses + learned_clauses
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"UNIT_PROPAGATION_BEGIN",
            f"READ_CLAUSES READ_BEGIN {format_list(clauses, is_var=True)} READ_END",
            f"READ_LEARNED_CLAUSES READ_BEGIN {format_list(learned_clauses, is_var=True)} READ_END",
            f"READ_ASSIGNMENTS READ_BEGIN {format_list(assignments_lst, is_var=True)} READ_END",
        ]
        self.current_unit_trace.extend(trace) 

    def on_unit_propagation_clause_propagation_loop_end(self, clause: list, all_assigned: bool, satisfied: bool,
                                                        conflict: bool, is_unit: Optional[bool] = None, 
                                                        learned_literal: Optional[int] = None, level: Optional[int] = None):
        trace = [
            f"CLAUSE {format_list(clause, is_var=True)}",
            f"ALL_ASSIGNED {str(int(all_assigned))}",
            f"SATISFIED {str(int(satisfied))}",
            f"CONFLICT {str(int(conflict))}",
            f"IS_UNIT {str(int(is_unit)) if is_unit is not None else 'None'}",
        ]
        if learned_literal is not None:
            trace = trace + [
                f"WRITE_ASSIGNMENTS WRITE_BEGIN {format_lit(learned_literal)} WRITE_END",
                f"WRITE_DECISION_LEVELS WRITE_BEGIN {encode_number_unary(level)} WRITE_END",
                f"WRITE_REASON_CLAUSES WRITE_BEGIN {format_list(clause, is_var=True)} WRITE_END",
            ]
        self.current_unit_trace.extend(trace) 

    def on_unit_propagation_loop_end(self, propagated: bool):
        trace = [
            f"PROPAGATED {str(int(propagated))}",
        ]
        self.current_unit_trace.extend(trace) 

    def on_unit_propagation_end(self, conflict_clause: Optional[list]):
        trace = []
        if conflict_clause is not None:
            trace = [f"WRITE_CONFLICT_CLAUSE WRITE_BEGIN {format_list(conflict_clause, is_var=True)} WRITE_END"]
        trace = trace + [f"UNIT_PROPAGATION_END"]
        self.current_unit_trace.extend(trace) 

        self.unit_propagation_traces.append(self.current_unit_trace)
        self.current_unit_trace = None


    ### SOLVE TRACES ###
    def on_solve_loop_start(self):
        trace = [
            f"CALL_UNIT_PROPAGATION",
        ]
        self.solve_trace.extend(trace) 

    def on_solve_conflict_found(self, level: int, is_unsat: bool):
        trace = [
            f"READ_LEVEL READ_BEGIN {encode_number_unary(level)} READ_END",
            f"IS UNSAT {str(int(is_unsat))}"
        ]
        if not is_unsat:
            trace = trace + [f"CALL_ANALYZE_CONFLICT"]
        else:
            trace = trace + [f"SOLVE_END"]

        self.solve_trace.extend(trace) 

    def on_solve_conflict_resolved(self):
        trace = [
            f"BACKTRACK",
        ]
        self.solve_trace.extend(trace)

    def on_solve_conflict_not_found(self, assignments: dict, n_vars: int, n_assigned_vars: int, is_sat: bool,
                                    new_lit: Optional[int] = None, level: Optional[int] = None):
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"READ_ASSIGNMENTS READ_BEGIN {format_list(assignments_lst, is_var=True)} READ_END",  
            f"N_VARS_ASSIGNED {encode_number_unary(n_assigned_vars)}",  
            f"N_VARS_TOTAL {encode_number_unary(n_vars)}",  
        ]
        if is_sat:
            trace = trace + ["SAT"] + ["SOLVE_END"]
        else:
            trace = trace + [
                f"WRITE_ASSIGNMENTS WRITE_BEGIN {format_lit(new_lit)} WRITE_END",
                f"WRITE_DECISION_LEVELS WRITE_BEGIN {encode_number_unary(level)} WRITE_END",
                f"WRITE_REASON_CLAUSES WRITE_BEGIN None WRITE_END",
                f"LEVEL_UP"
            ]
        self.solve_trace.extend(trace) 

        
    ### SOLVER START ###
    def on_start(self, clauses: list):
        trace = [
            f"SOLVE_BEGIN",
            f"READ_CLAUSES READ_BEGIN {format_list(clauses, is_var=True)} READ_END",
        ]
        self.input_clauses_trace = format_list(clauses, is_var=True)
        self.solve_trace.extend(trace) 

    def get_unit_propagation_trace(self, id: int):
        return self.unit_propagation_traces[id]

    def get_analyze_conflict_trace(self, id: int):
        return self.analyze_conflict_traces[id]

    def get_solve_trace_with_subcalls(self) -> list[str]:
        full_trace = []
        unit_idx = 0
        conflict_idx = 0

        def simplify_reads(trace):
            return [
                "READ_CLAUSES"
                if line.startswith("READ_CLAUSES") else line
                for line in trace
            ]

        for line in self.solve_trace:
            full_trace.append(line)
            if line == "CALL_UNIT_PROPAGATION":
                full_trace.extend(simplify_reads(self.unit_propagation_traces[unit_idx]))
                unit_idx += 1
            elif line == "CALL_ANALYZE_CONFLICT":
                full_trace.extend(self.analyze_conflict_traces[conflict_idx])
                conflict_idx += 1

        return full_trace

    def get_trace(self) -> list[str]:
        return {
            'input_clauses': self.input_clauses_trace,
            'solve_trace': self.solve_trace, 
            'solve_trace_with_subcalls': self.get_solve_trace_with_subcalls(), 
            'unit_prop_traces': self.unit_propagation_traces,
            'analyze_conflict_traces': self.analyze_conflict_traces
        }

def encode_number_unary(num: int) -> str:
    return ' '.join(['I'] * num)

def format_lit(literal: int) -> str:
    if literal < 0:
        return f"-x{abs(literal)}"
    return f"x{literal}"

def format_list(data, is_var=False, brackets=False, use_unary=False) -> str:
    # If it's just a single integer:
    if isinstance(data, int):
        if is_var:
            return format_lit(data) 
        elif use_unary:
            return encode_number_unary(data)
        else:
            return str(data)
    
    # If it's a list, recurse on each item:
    if isinstance(data, list):
        # Build each element’s string and join with spaces
        contents = " ".join(format_list(item, is_var=is_var, brackets=True, use_unary=use_unary) for item in data)
        if brackets:
            return f"[ {contents} ]"
        else:
            return contents
    
    # Fallback for types that aren’t int or list
    return str(data)