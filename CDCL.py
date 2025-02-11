from pysat.solvers import Glucose3
import random
import numpy as np
from tqdm import tqdm
import json

class CDCLSolver:
    def __init__(self, clauses):
        self.clauses = clauses
        self.assignments = {}
        self.level = 0
        self.decision_level = {}
        self.implication_graph = {}
        self.reason_clauses = {}
        self.learned_clauses = []
        self.trace = []
        
    def solve(self):
        self.trace.append(f"NUM_VARS {self.count_variables()}")  # number of variables
        while True:
            self.trace.append(f"ASG {format_dict_var(self.assignments)} LVL {self.level}")
            conflict = self.unit_propagate()
            if conflict:
                if self.level == 0:
                    self.trace.append("UNSAT")  # unsat
                    return False
                learned_clause = self.analyze_conflict(conflict)
                backtrack_level = self.find_backtrack_level(learned_clause)
                self.backtrack(backtrack_level)
                self.learned_clauses.append(learned_clause)
            else:
                self.trace.append(f'ASG_LEN {len(self.assignments)}') # assignment length
                
                if len(self.assignments) == self.count_variables():
                    self.trace.append("SAT")  # sat
                    return True
                var = self.pick_branching_variable()
                if var is None:
                    return True
                self.level += 1
                self.assign(var, True, None)
                self.trace.append(f'BRANCH => x{var} SET_TO + LVL {self.level}')  # variable x set to 1 at level lv by branching

    def unit_propagate(self,):
        while True:
            propagated = False
            for clause in self.clauses + self.learned_clauses:
                status, value = self.evaluate_clause(clause)
                if status and not value:
                    self.trace.append(f"CONFLICT {format_list_var(clause)}") # found conflic
                    return clause
                elif self.is_unit(clause):
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    value_char = ('+' if value else '-')
                    self.trace.append(f'UNIT {format_list_var(clause)} => x{var} SET_TO {value_char} LVL {self.level}') # variable x set to `value` at level lv by reason
                    self.assign(var, value, clause)
                    propagated = True
            if not propagated:
                break
        return None

    def analyze_conflict(self, conflict_clause):
        # Initialize sets to track variables at current decision level and literals for learned clause
        current_level_vars = set()  # Variables assigned at current decision level
        learned_lits = set()        # Literals that will form the learned clause

        # Start with literals from the conflict clause
        queue = self.get_literals_from_clause(conflict_clause) #
        self.trace.append(f"ANALYSIS_QUEUE {format_list_var(queue)}") #TODO
        while True:
            # Process each literal in the current clause
            for lit in queue:
                var = abs(lit)  # Get variable (removing sign)

                # If variable was assigned at current level, add to current_level_vars
                if self.decision_level.get(var) == self.level:
                    current_level_vars.add(var)
                # If assigned at earlier level, add to learned clause
                else:
                    learned_lits.add(-var if self.assignments[var] else var)
                    self.trace.append(f"LEARNED_LIT {format_lit(lit)}")
            
            # UIP condition: only one variable from current decision level remains
            if len(current_level_vars) <= 1:
                self.trace.append("UIP")
                break
                
            # Get most recently assigned variable from current level
            var = self.get_latest_assigned(current_level_vars)
            current_level_vars.remove(var)
            
            # Get the clause that caused this variable's assignment
            reason = self.reason_clauses.get(var)
            self.trace.append(f"UNIT {format_list_var(reason)} => x{var}") #TODO
            if reason:
                queue = [lit for lit in self.get_literals_from_clause(reason) 
                        if abs(lit) != var]
        
        # Create set of literals from current level variables with opposite polarity
        current_level_lits = {-var if self.assignments[var] else var for var in current_level_vars}
        new_clause = list(learned_lits.union(current_level_lits)) # TODO check if this is correct'
        self.trace.append(f"LEARNED_CLS {format_list_var(new_clause)}")
        return new_clause

    def backtrack(self, level):
        self.level = level
        self.assignments = {var: value for var, value in self.assignments.items() 
                          if self.decision_level[var] <= level}
        self.reason_clauses = {var: clause for var, clause in self.reason_clauses.items() 
                             if var in self.assignments}

    def assign(self, var, value, reason):
        self.assignments[var] = value
        self.decision_level[var] = self.level
        if reason:
            self.reason_clauses[var] = reason

    def evaluate_clause(self, clause):
        satisfied = False
        all_assigned = True
        for lit in clause:
            var = abs(lit)
            if var in self.assignments:
                if (lit > 0) == self.assignments[var]:
                    satisfied = True
                    break
            else:
                all_assigned = False
        return all_assigned, satisfied

    def is_unit(self, clause):
        unassigned = 0
        satisfied = False
        for lit in clause:
            var = abs(lit)
            if var in self.assignments:
                if (lit > 0) == self.assignments[var]:
                    satisfied = True
                    break
            else:
                unassigned += 1
        return not satisfied and unassigned == 1

    def get_unassigned_literal(self, clause):
        for lit in clause:
            if abs(lit) not in self.assignments:
                return lit
        return None

    def get_literals_from_clause(self, clause):
        return [lit for lit in clause]

    def get_latest_assigned(self, vars_set):
        return max(vars_set, key=lambda var: list(self.assignments.keys()).index(var))

    def find_backtrack_level(self, learned_clause):
        self.trace.append(f"FIND_BACKTRACK_LVL")
        levels = [self.decision_level[abs(lit)] for lit in learned_clause 
                 if abs(lit) in self.decision_level]
        self.trace.append(f"LEVELS {format_list_var(levels)}")
        if not levels:
            self.trace.append("GOTO_LVL 0")
            return 0
        levels.sort(reverse=True)
        goto = levels[1] if len(levels) > 1 else 0
        self.trace.append(f"GOTO_LVL {goto}")
        return goto

    def pick_branching_variable(self):
        for var in range(1, self.count_variables() + 1):
            if var not in self.assignments:
                return var
        return None

    def count_variables(self):
        vars_set = set()
        for clause in self.clauses:
            for lit in clause:
                vars_set.add(abs(lit))
        return len(vars_set)

def format_lit(lit: int) -> str:
    var = abs(lit)
    return f'+ x{var}' if lit > 0 else f'- x{var}'

def format_list_var(lst) -> str:
    if len(lst) == 0:
        return "[ ]"
    if type(lst) == list and type(lst[0]) == int:
        return "[ " + " ".join(map(format_lit, lst)) + " ]"
    return "[ " + ' '.join(map(str, lst)) + " ]"

def format_dict_var(dct) -> str:
    return format_list_var([var if val else -var for var, val in dct.items()])


def generate_random_formula(n_vars, n_clauses, clause_length=3):
    var_range = range(1, n_vars + 1)
    clauses = []
    for _ in range(n_clauses):
        clause_vars = random.sample(var_range, clause_length)
        clause = [var if random.random() < 0.5 else -var for var in clause_vars]
        clauses.append(clause)
    return clauses

 ######## TEST
balanced_n_clauses = {3: 19, 4: 24, 5: 28, 6: 33, 7: 37, 8: 41, 9: 45, 10: 50, 11: 54, 12: 58, 13: 63, 14: 67, 15: 71}
# n_vars_used_list_medium = [3, 4, 5, 6, 7, 8]
n_vars_used_list_hard = [9, 10, 11, 12, 13, 14, 15]
# n_vars_used_list_easy = [4] #  ... n_clauses is 24 in this case
variance = 2
formulas = []
for _ in tqdm(range(100000), desc="Generating formulas..."):
    n_var = random.choice(n_vars_used_list_hard)
    n_clauses = int(np.random.normal(loc=balanced_n_clauses[n_var], scale=variance))
    formulas.append(generate_random_formula(n_vars=n_var, n_clauses=n_clauses))

traces =  []
n_sat = 0
for clauses in tqdm(formulas, desc="Creating the traces..."):
    solver = CDCLSolver(clauses)
    is_satisfiable = solver.solve()
    n_sat = n_sat + is_satisfiable
    traces.append(solver.trace)

print(f'SAT ratio = {(n_sat / len(formulas)):.2f}')
variable_name_range = list(range(1, 20))

def remap_variables(data, variable_name_range):
    # Create a mapping for x0, x1, ..., xN to randomly chosen variable names
    mapping = {f"x{i}": f"y{random.choice(variable_name_range)}" for i in range(1, 21)}

    # Replace occurrences of each variable in the data
    for old_var in sorted(mapping.keys(), key=len, reverse=True):
        data = data.replace(old_var, mapping[old_var])

    return data.replace('y', 'x')


with open("cdcl_dataset.json", "w") as f:
    data_list = []
    for formula, trace in tqdm(zip(formulas, traces), desc="Writing into a file..."):
        trace_str = ' '.join(trace) 
        data = (
            "FORMULA_START "
            f"{format_list_var([format_list_var(clause) for clause in formula])} "
            "FORMULA_END " 
            "TRACE_START "
            f"{trace_str} "
            "TRACE_END"
        )
        remapped_data = remap_variables(data, variable_name_range)
        data_list.append({"text": remapped_data})
    
    json.dump(data_list, f, indent=4)
