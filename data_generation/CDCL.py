from pysat.solvers import Glucose3
import random
import argparse
import numpy as np
from tqdm import tqdm
from tracer import Tracer

class CDCLSolver:
    def __init__(self, clauses, tracer: Tracer):
        self.clauses = clauses
        self.assignments = {}
        self.level = 0
        self.decision_level = {}
        self.implication_graph = {}
        self.reason_clauses = {}
        self.learned_clauses = []
        self.tracer = tracer
        
    def solve(self):
        self.tracer.on_start(self.clauses)
        while True:
            self.tracer.on_solve_loop_start()
            conflict = self.unit_propagate()
            if conflict:
                if self.level == 0:
                    self.tracer.on_solve_conflict_found(self.level, True)
                    return False
                self.tracer.on_solve_conflict_found(self.level, False)
                learned_clause = self.analyze_conflict(conflict)
                backtrack_level = self.find_backtrack_level(learned_clause)
                self.tracer.on_solve_conflict_resolved()
                self.backtrack(backtrack_level)
                self.learned_clauses.append(learned_clause)
            else:
                n_vars, n_assigned_vars = self.count_variables(), len(self.assignments)
                if n_assigned_vars == n_vars:
                    self.tracer.on_solve_conflict_not_found(self.assignments, n_vars, n_assigned_vars, True)
                    return True
                var = self.pick_branching_variable()
                value = random.choice([True, False])
                lit = var if value else -var
                # self.assign(var, True, None)
                self.tracer.on_solve_conflict_not_found(self.assignments, n_vars, n_assigned_vars, False, lit)
                self.level += 1
                self.assign(var, value, None)

    def unit_propagate(self,):
        self.tracer.on_unit_propagation_start(self.clauses, self.learned_clauses, self.assignments)
        while True:
            propagated = False
            for clause in self.clauses + self.learned_clauses:
                status, value = self.evaluate_clause(clause)  # status = all_assigned, value = satisfied
                if status and not value:
                    self.tracer.on_unit_propagation_clause_propagation_loop_end(clause, status, value, True)  # conflict
                    self.tracer.on_unit_propagation_loop_end(propagated)
                    self.tracer.on_unit_propagation_end(clause)
                    return clause
                elif self.is_unit(clause):
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    self.assign(var, value, clause)
                    self.tracer.on_unit_propagation_clause_propagation_loop_end(clause, status, value, False, True, lit)
                    propagated = True
            self.tracer.on_unit_propagation_loop_end(propagated)
            if not propagated:
                break
        self.tracer.on_unit_propagation_end(None)
        return None

    def analyze_conflict(self, conflict_clause):
        self.tracer.on_analyze_conflict_start(
            self.assignments, self.decision_level, self.reason_clauses, conflict_clause, self.level
        )

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
                self.tracer.on_analyze_conflict_iteration_end(queue, current_level_vars - {var}, learned_lits, True)
                break
                
            # Get most recently assigned variable from current level
            var = self.get_latest_assigned(current_level_vars)
            current_level_vars.remove(var)
            
            # Get the clause that caused this variable's assignment
            reason = self.reason_clauses.get(var)
            self.tracer.on_analyze_conflict_iteration_end(queue, current_level_vars, learned_lits, False, var, reason)
            if reason:
                queue = [lit for lit in self.get_literals_from_clause(reason) 
                        if abs(lit) != var]
        
        # Create set of literals from current level variables with opposite polarity
        current_level_lits = {-var if self.assignments[var] else var for var in current_level_vars}
        new_clause = list(learned_lits.union(current_level_lits)) # TODO check if this is correct'
        self.tracer.on_analyze_conflict_end(new_clause)
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
            self.tracer.on_analyze_conflict_find_backtrack_level_end(learned_clause, 0)
            return 0
        levels.sort(reverse=True)
        goto = levels[1] if len(levels) > 1 else 0
        self.tracer.on_analyze_conflict_find_backtrack_level_end(learned_clause, goto, levels)
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

    def get_tracer(self):
        return self.tracer

def main(args):
    from generate_data import generate_random_formula

    formulas = []
    for _ in range(args.n_formulas):
        formulas.append(generate_random_formula(n_vars=args.n_vars))

    for clauses in formulas:
        g = Glucose3()
        for clause in clauses:
            g.add_clause(clause)
        is_sat_pysat = g.solve()
        g.delete()

        solver = CDCLSolver(clauses, Tracer())
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

    # print an example trace from CDCLSolver
    trace = solver.tracer.get_trace()
    print('\n'.join(trace['solve_trace_with_subcalls']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom CDCL Solver with generated SAT formulas.")
    parser.add_argument(
        "--n_formulas", type=int, default=50,
        help="Number of formulas to generate (default: 50)."
    )
    parser.add_argument(
        "--n_vars", type=int, default=10,
        help="Number of variables in the generated formulas (default: 10)."
    )
    args = parser.parse_args()
    main(args)