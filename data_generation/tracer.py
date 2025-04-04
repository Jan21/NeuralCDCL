from typing import Optional

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


    ### ANALYZE_CONFLICT TRACES ###
    def on_analyze_conflict_start(self, assignments: dict, decision_level: dict, reason_clauses: dict, 
                                  conflict_clause: list, level: int):
        self.current_analyze_trace = []
        decision_level_adj = [decision_level[var] for var in assignments.keys()]
        reason_clauses_adj = [reason_clauses[var] if var in reason_clauses else [] for var in assignments.keys()]
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"READ ASSIGNMENTS {format_list(assignments_lst, is_var=True)}",
            f"READ DECISION_LEVELS {format_list(decision_level_adj, is_var=False)}",
            f"READ REASON_CLAUSES {format_list(reason_clauses_adj, is_var=True)}",
            f"READ CONFLICT_CLAUSE {format_list(conflict_clause, is_var=True)}",
            f"READ LEVEL {level}\n",
        ]
        trace = '\n'.join(trace)
        self.current_analyze_trace.append(trace) 

    def on_analyze_conflict_iteration_end(self, queue: list, curr_level_vars: set, learned_lits: set, 
                                          is_uip: bool, selected_var: Optional[int] = None, reason_clause: Optional[list] = None):
        trace = [
            f"QUEUE {format_list(queue, is_var=True)}",
            f"RESOLVING {format_lit(selected_var) if selected_var is not None else 'None'}",
            f"REASON_CLAUSE {format_list(reason_clause, is_var=True) if reason_clause is not None else 'None'}"
            f"CURRENT_LVL_VARS {format_list(list(curr_level_vars), is_var=True)}",
            f"LEARNED_LITS {format_list(list(learned_lits), is_var=True)}",
            f"IS_UIP {str(int(is_uip))}\n",
        ]
        self.current_analyze_trace.append('\n'.join(trace)) 

    def on_analyze_conflict_end(self, new_clause: list):
        trace = [
            f"WRITE NEW_CLAUSE {format_list(new_clause, is_var=True)}\n",
        ]
        self.current_analyze_trace.append('\n'.join(trace)) 

    def on_analyze_conflict_find_backtrack_level_end(self, learned_clause: list, goto_level: int, levels: Optional[list] = None):
        trace = [
            f"DECISION_LEVELS {format_list(levels if levels else [], is_var=False)}",
            f"WRITE BACKTRACK_LEVEL {goto_level}",
            f"END"
        ]
        self.current_analyze_trace.append('\n'.join(trace)) 

        self.analyze_conflict_traces.append(self.current_analyze_trace)
        self.current_analyze_trace = None


    ### UNIT_PROPAGATION TRACES ###
    def on_unit_propagation_start(self, clauses: list, learned_clauses: list, assignments: dict):
        self.current_unit_trace = []
        clauses_combined = clauses + learned_clauses
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"READ CLAUSES {format_list(clauses, is_var=True)}",
            f"READ LEARNED_CLAUSES {format_list(learned_clauses, is_var=True)}",
            f"READ ASSIGNMENTS {format_list(assignments_lst, is_var=True)}\n",
        ]
        trace = '\n'.join(trace)
        self.current_unit_trace.append(trace) 

    def on_unit_propagation_clause_propagation_loop_end(self, clause: list, all_assigned: bool, satisfied: bool,
                                                        conflict: bool, is_unit: Optional[bool] = None, 
                                                        learned_literal: Optional[int] = None):
        trace = [
            f"CLAUSE {format_list(clause, is_var=True)}",
            f"ALL_ASSIGNED {str(int(all_assigned))}",
            f"SATISFIED {str(int(satisfied))}",
            f"CONFLICT {str(int(conflict))}",
            f"IS_UNIT {str(int(is_unit)) if is_unit is not None else 'None'}",
        ]
        if learned_literal is not None:
            trace = trace + [
                f"WRITE LIT {format_lit(learned_literal)} REASON {format_list(clause, is_var=True)}\n",
            ]
        self.current_unit_trace.append('\n'.join(trace)) 

    def on_unit_propagation_loop_end(self, propagated: bool):
        trace = [
            f"PROPAGATED {str(int(propagated))}",
        ]
        self.current_unit_trace.append('\n'.join(trace)) 

    def on_unit_propagation_end(self, conflict_clause: Optional[list]):
        trace = []
        if conflict_clause is not None:
            trace = [f"WRITE CONFLICT_CLAUSE {format_list(conflict_clause, is_var=True)}"]
        trace = trace + [f"END"]
        trace = '\n'.join(trace)
        self.current_unit_trace.append(trace) 

        self.unit_propagation_traces.append(self.current_unit_trace)
        self.current_unit_trace = None


    ### SOLVE TRACES ###
    def on_solve_loop_start(self):
        trace = [
            f"CALL UNIT_PROPAGATION",
        ]
        self.solve_trace.append('\n'.join(trace)) 

    def on_solve_conflict_found(self, level: int, is_unsat: bool):
        trace = [
            f"READ LEVEL {level}",
            f"IS UNSAT {str(int(is_unsat))}"
        ]
        if not is_unsat:
            trace = trace + [f"CALL ANALYZE_CONFLICT"]
        else:
            trace = trace + [f"END"]

        self.solve_trace.append('\n'.join(trace)) 

    def on_solve_conflict_resolved(self):
        trace = [
            f"BACKTRACK",
        ]
        self.solve_trace.append('\n'.join(trace)) 

    def on_solve_conflict_not_found(self, assignments: dict, n_vars: int, n_assigned_vars: int, is_sat: bool, new_lit: Optional[int] = None):
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"READ ASSIGNMENTS {format_list(assignments_lst, is_var=True)}",  
            f"N_VARS_ASSIGNED {n_assigned_vars}",  
            f"N_VARS_TOTAL {n_vars}",  
        ]
        if is_sat:
            trace = trace + ["SAT"] + ["END"]
        else:
            trace = trace + [
                f"WRITE LIT {new_lit} REASON None",
            ]
        self.solve_trace.append('\n'.join(trace)) 

        
    ### SOLVER START ###
    def on_start(self, clauses: list):
        trace = [
            f"FORMULA {format_list(clauses, is_var=True)}",
        ]
        self.solve_trace.append('\n'.join(trace)) 

    def get_unit_propagation_trace(self, id: int):
        return self.unit_propagation_traces[id]

    def get_analyze_conflict_trace(self, id: int):
        return self.unit_propagation_traces[id]

    def get_solve_trace_with_subcalls(self) -> list[str]:
        full_trace = []
        unit_idx = 0
        conflict_idx = 0
        for line in self.solve_trace:
            full_trace.append(line)
            if line == "CALL UNIT_PROPAGATION":
                full_trace.extend(self.unit_propagation_traces[unit_idx])
                unit_idx += 1
            elif line == "CALL ANALYZE_CONFLICT":
                full_trace.extend(self.analyze_conflict_traces[conflict_idx])
                conflict_idx += 1
        return full_trace

    def get_trace(self) -> list[str]:
        return {
            'solve_trace': self.solve_trace, 
            'solve_trace_with_subcalls': self.get_solve_trace_with_subcalls(), 
            'unit_prop_traces': self.unit_propagation_traces,
            'analyze_conflict_traces': self.analyze_conflict_traces
        }

def format_lit(literal: int) -> str:
    if literal < 0:
        return f"-x{abs(literal)}"
    return f"x{literal}"

def format_list(data, is_var=False) -> str:
    # If it's just a single integer:
    if isinstance(data, int):
        return format_lit(data) if is_var else str(data)
    
    # If it's a list, recurse on each item:
    if isinstance(data, list):
        # Build each element’s string and join with spaces
        contents = " ".join(format_list(item, is_var=is_var) for item in data)
        return f"[{contents}]"
    
    # Fallback for types that aren’t int or list
    return str(data)