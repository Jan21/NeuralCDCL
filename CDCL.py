from pysat.solvers import Glucose3
import random

class CDCLSolver:
    def __init__(self, clauses):
        self.clauses = clauses
        self.assignments = {}
        self.level = 0
        self.decision_level = {}
        self.implication_graph = {}
        self.reason_clauses = {}
        self.learned_clauses = []
        
    def solve(self):
        while True:
            conflict = self.unit_propagate()
            if conflict:
                if self.level == 0:
                    return False
                learned_clause = self.analyze_conflict(conflict)
                backtrack_level = self.find_backtrack_level(learned_clause)
                self.backtrack(backtrack_level)
                self.learned_clauses.append(learned_clause)
            else:
                if len(self.assignments) == self.count_variables():
                    return True
                var = self.pick_branching_variable()
                if var is None:
                    return True
                self.level += 1
                self.assign(var, True, None)

    def unit_propagate(self):
        while True:
            propagated = False
            for clause in self.clauses + self.learned_clauses:
                status, value = self.evaluate_clause(clause)
                if status and not value:
                    return clause
                elif self.is_unit(clause):
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
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
            
            # UIP condition: only one variable from current decision level remains
            if len(current_level_vars) <= 1:
                break
                
            # Get most recently assigned variable from current level
            var = self.get_latest_assigned(current_level_vars)
            current_level_vars.remove(var)
            
            # Get the clause that caused this variable's assignment
            reason = self.reason_clauses.get(var)
            if reason:
                queue = [lit for lit in self.get_literals_from_clause(reason) 
                        if abs(lit) != var]
        
        # Create set of literals from current level variables with opposite polarity
        current_level_lits = {-var if self.assignments[var] else var for var in current_level_vars}
        new_clause =list(learned_lits.union(current_level_lits)) # TODO check if this is correct'
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
        levels = [self.decision_level[abs(lit)] for lit in learned_clause 
                 if abs(lit) in self.decision_level]
        if not levels:
            return 0
        levels.sort(reverse=True)
        return levels[1] if len(levels) > 1 else 0

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
formulas = []
for i in range(50):
    formulas.append(generate_random_formula(50))

for clauses in formulas:
    g = Glucose3()
    for clause in clauses:
        g.add_clause(clause)
    is_sat_pysat = g.solve()
    g.delete()


    solver = CDCLSolver(clauses)
    is_satisfiable = solver.solve()
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
