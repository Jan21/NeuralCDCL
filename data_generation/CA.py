from pysat.solvers import Glucose3
import random
from tqdm import tqdm
from typing import Optional

def log_var(var):
    return f"x{var}"

def log_decision_level(decision_level):
    string = ''
    for var, level in decision_level.items():
        string += f"x{var} = {level} , "
    return string[:-3]

def log_assignments(assignments):
    sorted_assignments = sorted(assignments.items(), key=lambda x: x[0])
    assignments_str = ""
    for var, value in sorted_assignments:
        assignments_str += f"x{var} = {value} , "
    return assignments_str[:-3]

def log_clause(clause,i):
    string = f'( '
    for l in clause:
        if l > 0:
            string += f"+ x{abs(l)} "
        else:
            string += f"- x{abs(l)} "
    return string.strip() + f' ) : {i}'

def log_new_clause(clause):
    string = f'( '
    for l in clause:
        if l > 0:
            string += f"+ x{abs(l)} "
        else:
            string += f"- x{abs(l)} "
    return string.strip() + f' )'


def log_clause_list(clause_list,clause2id):
    string = ''
    for clause in clause_list:
        i = clause2id[tuple(clause)]
        string += log_clause(clause,i) + ' , '
    return string[:-3]

def log_current_level_vars(current_level_vars):
    string = ''
    for var in current_level_vars:
        string += f"{log_var(var)} , "
    return string[:-3]

def log_queue(queue):
    string = ''
    for lit in queue:
        var = abs(lit)
        if lit > 0:
            string += f"+ x{var} , "
        else:
            string += f"- x{var} , "
    return string[:-3]

def log_reasons(clauses,clause2id):
    string = ''
    for k,v in clauses.items():
        string += f"{log_var(k)} -> {clause2id[tuple(v)]} , "
    return string[:-3]


class CDCLSolver:
    def __init__(self, clauses):
        self.clauses = clauses
        self.clause2id = {tuple(c):f'c {i}' for i,c in enumerate(clauses)}
        self.assignments = {}
        self.level = 0
        self.decision_level = {}
        self.implication_graph = {}
        self.reason_clauses = {}
        self.learned_clauses = []
        self.trace = []
        
    def solve(self):
        #self.trace.append(f"num vars: {self.count_variables()}")
        while True:
            #self.trace.append(f"current level: {self.level}")
            #self.trace.append(f'current assignment: {self.assignments}') #TODO
            conflict = self.unit_propagate()
            if conflict:
                if self.level == 0:
                    #self.trace.append("UNSAT")
                    return False
                learned_clause = self.analyze_conflict(conflict)
                backtrack_level = self.find_backtrack_level(learned_clause)
                self.backtrack(backtrack_level)
                self.learned_clauses.append(learned_clause)
                self.clause2id[tuple(learned_clause)] = f'c{len(self.clause2id)}'
            else:
                #self.trace.append(f'assignment length: {len(self.assignments)}')
                
                if len(self.assignments) == self.count_variables():
                    #self.trace.append("SAT")
                    return True
                var = self.pick_branching_variable()
                if var is None:
                    return True
                self.level += 1
                self.assign(var, True, None)
                #self.trace.append(f'variable assigned: x{var} = {True} at level {self.level} by branching')

    def unit_propagate(self,):
        #self.trace.append("UP begin")
        while True:
            #self.trace.append('UP iteration')
            propagated = False
            for clause in self.clauses + self.learned_clauses:
                status, value = self.evaluate_clause(clause)
                if status and not value:
                    #self.trace.append(f"found conflict: {clause}") #TODO
                    return clause
                elif self.is_unit(clause):
                    #self.trace.append(f'unit found: {clause}') # TODO convert clause
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    #self.trace.append(f'variable assigned: x{var} = {value} at level {self.level} because of reason {clause}') #TODO
                    self.assign(var, value, clause)
                    propagated = True
            if not propagated:
                #self.trace.append("nothing propagated")
                break
        return None



    def analyze_conflict(self, conflict_clause):
        trace = []
        
        trace.append(f"assignments: {log_assignments(self.assignments)}")
        trace.append(f"clauses: {log_clause_list(self.clauses + self.learned_clauses,self.clause2id)}")
        trace.append(f"decision-level: {log_decision_level(self.decision_level)}")
        trace.append(f"reason-clauses: {log_reasons(self.reason_clauses,self.clause2id)}")
        trace.append(f"conflict-clause: {log_clause(conflict_clause,self.clause2id[tuple(conflict_clause)])}")
        trace.append("AC-begin")
        # Initialize sets to track variables at current decision level and literals for learned clause
        current_level_vars = set()  # Variables assigned at current decision level
        learned_lits = set()        # Literals that will form the learned clause

        # Start with literals from the conflict clause
        queue = self.get_literals_from_clause(conflict_clause) #
        trace.append(f"queue: {log_queue(queue)}")
        while True:
            # Process each literal in the current clause
            for lit in queue:
                var = abs(lit)  # Get variable (removing sign)
                trace.append(f"checking-variable: {log_var(var)} at {self.decision_level.get(var)}")
                
                # If variable was assigned at current level, add to current_level_vars
                if self.decision_level.get(var) == self.level:
                    current_level_vars.add(var)
                    trace.append(f"current-level-vars: {log_current_level_vars(current_level_vars)}") #TODO
                # If assigned at earlier level, add to learned clause
                else:
                    learned_lits.add(-var if self.assignments[var] else var)
                    var_str = log_var(var)
                    trace.append(f"learned-lits: { "- " + var_str if self.assignments[var] else "+ " + var_str}") #TODO
            
            # UIP condition: only one variable from current decision level remains
            if len(current_level_vars) <= 1:
                trace.append("UIP")
                break
 
                
            # Get most recently assigned variable from current level
            var = self.get_latest_assigned(current_level_vars)
            trace.append(f"latest-assigned: {log_var(var)}")
            current_level_vars.remove(var)
            
            # Get the clause that caused this variable's assignment
            reason = self.reason_clauses.get(var)
            trace.append(f"reason-for {log_var(var)} is {self.clause2id[tuple(reason)]}") #TODO
            if reason:
                queue = [lit for lit in self.get_literals_from_clause(reason) 
                        if abs(lit) != var]
                trace.append(f"queue: {log_queue(queue)}") # TODO
        
        # Create set of literals from current level variables with opposite polarity
        current_level_lits = [-var if self.assignments[var] else var for var in current_level_vars]
        trace.append(f"current-level-lits: {log_queue(current_level_lits)}")
        new_clause =list(learned_lits.union(set(current_level_lits))) # TODO check if this is correct'
        trace.append(f"new-clause: {log_new_clause(new_clause)}")
        trace.append("AC-end")
        self.trace.append(trace)
        return new_clause

    def backtrack(self, level):
        #self.trace.append("BT begin")
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
        #self.trace.append(f"FB begin")
        levels = [self.decision_level[abs(lit)] for lit in learned_clause 
                 if abs(lit) in self.decision_level]
        #self.trace.append(f"levels: {levels}")
        if not levels:
            self.trace.append("GOTO level: 0")
            return 0
        levels.sort(reverse=True)
        goto = levels[1] if len(levels) > 1 else 0
        #self.trace.append(f"GOTO level: {goto}")
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

def generate_random_formula(n_vars: int, clause_length: int = 3, variance: float = 0.1, n_clauses: Optional[int] = None):
    """
    Generates a random CNF formula.
    
    Args:
        n_vars (int): Number of variables.
        n_clauses (int, optional): Fixed number of clauses. If None, it will be estimated.
        clause_length (int): Number of literals per clause.
        variance (float): Relative standard deviation in clause count (e.g., 0.1 = ±10%).

    Returns:
        list[list[int]]: A list of clauses, each clause is a list of literals.
    """
    balanced_n_clauses = {3: 19, 4: 24, 5: 28, 6: 33, 7: 37, 8: 41, 9: 45, 10: 50, 11: 54, 12: 58, 
                          13: 63, 14: 67, 15: 71, 16: 76, 17: 79, 18: 83, 19: 87, 20: 92}
    if n_clauses == None:
        base = balanced_n_clauses.get(n_vars, int(n_vars * 4.26))
        delta = int(base * variance)
        n_clauses = random.randint(base - delta, base + delta)

    var_range = range(1, n_vars + 1)
    clauses = []
    for _ in range(n_clauses):
        clause_vars = random.sample(var_range, clause_length)
        clause = [var if random.random() < 0.5 else -var for var in clause_vars]
        clauses.append(clause)
    return clauses

 ######## TEST
formulas = []
for i in tqdm(range(100000)):
    num_vars = random.randint(5, 15)
    formulas.append(generate_random_formula(num_vars))


traces =  []
for clauses in tqdm(formulas):
    g = Glucose3()
    for clause in clauses:
        g.add_clause(clause)
    is_sat_pysat = g.solve()
    g.delete()


    solver = CDCLSolver(clauses)
    is_satisfiable = solver.solve()
    traces += solver.trace
    assert is_sat_pysat == is_satisfiable

    if is_satisfiable:
    # Verify CDCL solution using PySAT
        g_verify = Glucose3()
        for clause in clauses:
            g_verify.add_clause(clause)
        assumptions = []
        for var, value in solver.assignments.items():
            assumptions.append(var if value else -var)
        
    # Check if solution satisfies formula
        is_valid = g_verify.solve(assumptions=assumptions)
        assert is_valid
        g_verify.delete()

trace_strs = []
for trace in traces:
    trace_str = " ; ".join(trace)
    trace_strs.append(trace_str)

trace_strs[0]


num_traces = len(trace_strs)
split_idx = int(0.90 * num_traces)

train_traces = trace_strs[:split_idx]
test_traces = trace_strs[split_idx:]

print(f"Split {num_traces} traces into {len(train_traces)} training and {len(test_traces)} testing traces")

# Save the splits to files
import json

# Convert traces to the required format (dict with "text" key)
train_data = [{"text": trace} for trace in train_traces]
test_data = [{"text": trace} for trace in test_traces]

# Save to JSON files
with open('train_CA.json', 'w') as f:
    json.dump(train_data, f)

with open('test_CA.json', 'w') as f:
    json.dump(test_data, f)

print(f"Saved training traces to train_traces.json")
print(f"Saved testing traces to test_traces.json")