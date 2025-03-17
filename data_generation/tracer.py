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
            f"ANALYZE_CONFLICT_ARGUMENTS",
            f"ASSIGNMENTS {format_list(assignments_lst, is_var=True)}",
            f"DECISION_LEVELS {format_list(decision_level_adj, is_var=False)}",
            f"REASON_CLAUSES {format_list(reason_clauses_adj, is_var=True)}",
            f"CONFLICT_CLAUSE {format_list(conflict_clause, is_var=True)}",
            f"LEVEL {level}",
            f"ANALYZE_CONFLICT_START",
        ]
        trace = ' '.join(trace)
        self.current_analyze_trace.append(trace) 
        self.solve_trace.append(trace)

    def on_analyze_conflict_iteration_end(self, queue: list, curr_level_vars_pre: set, learned_lits: set, 
                                          is_uip: bool, selected_var: Optional[int] = None, curr_level_vars_pos: Optional[set] = None, 
                                          reason_clause: Optional[list] = None):
        trace = [
            f"QUEUE {format_list(queue, is_var=True)}",
            f"CURRENT_LEVEL_VARS_PRE {format_list(list(curr_level_vars_pre), is_var=True)}",
            f"LEARNED_LITERALS {format_list(list(learned_lits), is_var=True)}",
            f"IS_UIP {str(int(is_uip))}",
        ]
        if selected_var and curr_level_vars_pos and reason_clause:
            trace = trace + [
                f"SELECTED_VAR {format_lit(selected_var)}",
                f"CURRENT_LEVEL_VARS_POS {format_list(list(curr_level_vars_pos), is_var=True)}",
                f"REASON_CLAUSE {format_list(reason_clause, is_var=True)}"
            ]
        self.current_analyze_trace.append(' '.join(trace)) 

    def on_analyze_conflict_end(self, curr_level_lits: set):
        trace = [
            f"CURRENT_LEVEL_LITERALS {format_list(list(curr_level_lits), is_var=True)}",
        ]
        self.current_analyze_trace.append(' '.join(trace)) 

    def on_analyze_conflict_find_backtrack_level_end(self, learned_clause: list, goto_level: int, levels: Optional[list] = None):
        trace = [
            f"LEVELS {format_list(levels if levels else [], is_var=False)}",
            f"ANALYZE_CONFLICT_RESULTS",
            f"LEARNED_CLAUSE {format_list(learned_clause, is_var=True)}",
            f"GOTO_LEVEL {goto_level}",
            f"ANALYZE_CONFLICT_END"
        ]
        self.current_analyze_trace.append(' '.join(trace)) 

        self.analyze_conflict_traces.append(self.current_analyze_trace)
        self.current_analyze_trace = None
        self.solve_trace.append(' '.join(trace[1:]))


    ### UNIT_PROPAGATION TRACES ###
    def on_unit_propagation_start(self, clauses: list, learned_clauses: list, assignments: dict):
        self.current_unit_trace = []
        clauses_combined = clauses + learned_clauses
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"UNIT_PROPAGATION_ARGUMENTS",
            f"CLAUSES {format_list(clauses_combined, is_var=True)}",
            f"ASSIGNMENTS {format_list(assignments_lst, is_var=True)}",
            f"UNIT_PROPAGATION_START"
        ]
        trace = ' '.join(trace)
        self.current_unit_trace.append(trace) 
        self.solve_trace.append(trace)

    def on_unit_propagation_clause_propagation_loop_end(self, clause: list, all_assigned: bool, satisfied: bool,
                                                        conflict: bool, is_unit: Optional[bool] = None, 
                                                        learned_literal: Optional[int] = None):
        trace = [
            f"CLAUSE {format_list(clause, is_var=True)}",
            f"ALL_ASSIGNED {str(int(all_assigned))}",
            f"SATISFIED {str(int(satisfied))}",
            f"CONFLICT {str(int(conflict))}",
        ]
        if is_unit and learned_literal:
            trace = trace + [
                f"IS_UNIT {str(int(is_unit))}",
                f"LEARNED_LITERAL {format_lit(learned_literal)}",
            ]
        self.current_unit_trace.append(' '.join(trace)) 

    def on_unit_propagation_loop_end(self, propagated: bool):
        trace = [
            f"PROPAGATED {str(int(propagated))}",
        ]
        self.current_unit_trace.append(' '.join(trace)) 

    def on_unit_propagation_end(self, assignments: dict):
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        trace = [
            f"UNIT_PROPAGATION_RESULTS",
            f"ASSIGNMENTS {format_list(assignments_lst, is_var=True)}",
            f"UNIT_PROPAGATION_END"
        ]
        trace = ' '.join(trace)
        self.current_unit_trace.append(trace) 

        self.unit_propagation_traces.append(self.current_unit_trace)
        self.current_unit_trace = None
        self.solve_trace.append(trace)


    ### SOLVE TRACES ###
    def on_solve_loop_start(self, assignments: dict, decision_level: dict, reason_clauses: dict, level: int):
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        decision_level_adj = [decision_level[var] for var in assignments.keys()]
        reason_clauses_adj = [reason_clauses[var] if var in reason_clauses else [] for var in assignments.keys()]
        trace = [
            f"ASSIGNMENTS {format_list(assignments_lst, is_var=True)}",
            f"DECISION_LEVELS {format_list(decision_level_adj, is_var=False)}",
            f"REASON_CLAUSES {format_list(reason_clauses_adj, is_var=True)}",
            f"LEVEL {level}"
        ]
        self.solve_trace.append(' '.join(trace)) 

    def on_solve_conflict_found(self, assignments: dict, decision_level: dict, reason_clauses: dict, conflict_clause: list, 
                                level: int, is_unsat: bool, learned_clause: Optional[list] = None, backtrack_level: Optional[int] = None):
        assignments_lst = [var * (-1 if val < 1 else 1) for var, val in assignments.items()]
        decision_level_adj = [decision_level[var] for var in assignments.keys()]
        reason_clauses_adj = [reason_clauses[var] if var in reason_clauses else [] for var in assignments.keys()]
        trace = [
            f"ASSIGNMENTS {format_list(assignments_lst, is_var=True)}",
            f"DECISION_LEVELS {format_list(decision_level_adj, is_var=False)}",
            f"REASON_CLAUSES {format_list(reason_clauses_adj, is_var=True)}",
            f"CONFLICT_CLAUSE {format_list(conflict_clause, is_var=True)}",
            f"LEVEL {level}"
        ]
        if is_unsat:
            trace = trace + ["UNSAT"] + ["SOLVE_END"]
        else:
            trace = trace + [
                f"LEARNED_CLAUSE {format_list(learned_clause, is_var=True)}",
                f"BACKTRACK_LEVEL {backtrack_level}",
            ]
        self.solve_trace.append(' '.join(trace)) 

    def on_solve_conflict_not_found(self, n_vars: int, n_assigned_vars: int, is_sat: bool, selected_var: Optional[int] = None):
        trace = [
            f"N_VARS {n_vars}",  
            f"N_ASSIGNED_VARS {n_assigned_vars}",  
        ]
        if is_sat:
            trace = trace + ["SAT"] + ["SOLVE_END"]
        else:
            trace = trace + [
                f"SELECTED_VAR {selected_var}",
            ]
        self.solve_trace.append(' '.join(trace)) 

        
    ### SOLVER START ###
    def on_start(self, clauses: list):
        trace = [
            f"SOLVE_ARGUMENTS",
            f"CLAUSES {format_list(clauses, is_var=True)}",
            f"SOLVE_START",
        ]
        self.solve_trace.append(' '.join(trace)) 

    def get_unit_propagation_trace(self, id: int):
        return self.unit_propagation_traces[id]

    def get_analyze_conflict_trace(self, id: int):
        return self.unit_propagation_traces[id]

    def get_solve_trace(self, packed: bool = True) -> list[str]:
        """
        If 'packed=True', return exactly what we appended in solve_trace.
        That means submodules appear only as '..._START' lines plus 
        '..._RESULTS' lines (the "header" + the final result).

        If 'packed=False', we expand each submodule’s internal lines (i.e. 
        everything *between* [MODULE_NAME]_START and [MODULE_NAME]_RESULTS) 
        directly into the solve trace.

        The returned value is a list of lines. You can always do '\n'.join(...) 
        if you want it as a single string.
        """
        if packed:
            return self.solve_trace

        expanded = []
        up_index = 0
        ac_index = 0
        for line in self.solve_trace:
            expanded.append(line)
            # Check if this line indicates a submodule start
            if "UNIT_PROPAGATION_START" in line:
                sub_lines = self.unit_propagation_traces[up_index]
                expanded_sub = self._extract_submodule_lines(
                    sub_lines, "UNIT_PROPAGATION_START", "UNIT_PROPAGATION_RESULTS"
                )
                expanded.extend(expanded_sub)
                up_index += 1

            elif "ANALYZE_CONFLICT_START" in line:
                sub_lines = self.analyze_conflict_traces[ac_index]
                expanded_sub = self._extract_submodule_lines(
                    sub_lines, "ANALYZE_CONFLICT_START", "ANALYZE_CONFLICT_RESULTS"
                )
                expanded.extend(expanded_sub)
                ac_index += 1

        return expanded

    def _extract_submodule_lines(self, sub_trace: list[str], start_token: str, end_token: str) -> list[str]:
        """
        Given an entire submodule trace, return only the lines that appear
        strictly after [MODULE_NAME]_START and strictly before [MODULE_NAME]_RESULTS,
        excluding those tokens themselves.
        """
        output = []
        in_logic = False

        for line in sub_trace:
            if start_token in line:
                # Once we see the start token, we start collecting lines 
                # that come after it
                in_logic = True
                continue

            if end_token in line:
                # Once we see the results token, we stop
                break

            if in_logic:
                output.append(line)

        return output

    def get_trace(self, packed: bool = True) -> list[str]:
        if packed:
            return {'solve_traces': [self.solve_trace], 'unit_prop_traces': self.unit_propagation_traces,
                    'analyze_conflict_traces': self.analyze_conflict_traces}
        return self.get_solve_trace(packed=False)

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