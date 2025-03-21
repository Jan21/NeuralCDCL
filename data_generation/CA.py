from pysat.solvers import Glucose3
import random
from tqdm import tqdm


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
        current_level_lits = {-var if self.assignments[var] else var for var in current_level_vars}
        trace.append(f"current-level-lits: {log_queue(current_level_lits)}")
        new_clause =list(learned_lits.union(current_level_lits)) # TODO check if this is correct'
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

def generate_random_formula(n_vars, n_clauses=None, clause_length=3):
    """
    Generate a random SAT formula with n variables.
    Args:
        n_vars: Number of variables
        n_clauses: Number of clauses (default: around 4.2 * n_vars for balanced SAT/UNSAT)
        clause_length: Length of each clause (default: 3 for 3-SAT)
    Returns:
        List of clauses, where each clause is a list of integers
    """
    # Use empirically determined ratio for balanced SAT/UNSAT
    if n_clauses is None:
        n_clauses = int(4.2 * n_vars)
        
    clauses = []
    for _ in range(n_clauses):
        # Generate a clause with random literals
        clause = []
        vars_used = set()
        
        while len(clause) < clause_length:
            # Pick a random variable that hasn't been used in this clause
            var = random.randint(1, n_vars)
            if var not in vars_used:
                # Randomly choose positive or negative literal
                lit = var if random.random() < 0.5 else -var
                clause.append(lit)
                vars_used.add(var)
                
        clauses.append(clause)
        
    return clauses

 ######## TEST
num_vars = 15
formulas = []
for i in tqdm(range(100000)):
    formulas.append(generate_random_formula(num_vars))


traces =  []
for clauses in formulas:
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

# Save traces to a pickle file
import pickle
import os

# Create directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save the traces to a pickle file
with open('data/traces.pkl', 'wb') as f:
    pickle.dump(traces, f)

print(f"Saved {len(traces)} traces to data/traces.pkl")
