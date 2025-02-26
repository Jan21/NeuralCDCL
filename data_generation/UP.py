from pysat.solvers import Glucose3
import random
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
import os

def log_clauses(clauses):
    string = 'clauses: '
    for c in clauses:
        string += "( "
        for l in c:
            if l > 0:
                string += f"+ {abs(l)} "
            else:
                string += f"- {abs(l)} "
        string += ") "
    string = string.strip()
    return string

def log_clauses(clauses):
    
    litdic = defaultdict(list)
    clausedic = defaultdict(list)
    for i,cl in enumerate(clauses):
        for l in cl:
            litdic[l].append(i+1)
            clausedic[i+1].append(l)
    
    token_positions = []
    
    # Add litdic lists
    for lit, clauses in litdic.items():
        token_positions.append(clauses)
        
    # Add clausedic lists 
    for clause_num, lits in clausedic.items():
        token_positions.append(lits)
    # Find maximum length of lists in result
    max_length = max(len(lst) for lst in token_positions)
    lit_dic_np = {}
    for k,v in litdic.items():
        padded_v = v + [0] * (max_length - len(v))
        lit_dic_np[k] = np.array(padded_v)

    clause_dic_np = {}
    for k,v in clausedic.items():
        padded_v = v + [0] * (max_length - len(v))
        clause_dic_np[k] = np.array(padded_v)

        
    return (clause_dic_np, lit_dic_np)

def log_clause(clause,i):
    string = f'( '
    for l in clause:
        if l > 0:
            string += f"+ x{abs(l)} "
        else:
            string += f"- x{abs(l)} "
    return string.strip() + ' ) : c{i}'

def log_clause_list(clause_list):
    string = ''
    for i,clause in enumerate(clause_list):
        string += log_clause(clause,i) + ' , '
    return string[:-3]

def log_assignments(assignments):
    sorted_assignments = sorted(assignments.items(), key=lambda x: x[0])
    assignments_str = ""
    for var, value in sorted_assignments:
        assignments_str += f"x{var} = {value} , "
    return assignments_str[:-3]

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

    def unit_propagate(self,):
        self.trace.append(f'assignments: {log_assignments(self.assignments)}')
        self.trace.append(f'clauses [ {log_clause_list(self.clauses)} ]')
        self.trace.append("UP begin")
        while True:
            self.trace.append('UP iteration')
            propagated = False
            for i,clause in enumerate(self.clauses + self.learned_clauses):
                status, value = self.evaluate_clause(clause)
                if status and not value:
                    self.trace.append(f"found conflict: c{i}") #{log_clause(clause,i)}") #TODO
                    #self.trace.append('UP end')
                    propagated = False
                    return clause
                elif self.is_unit(clause):
                    self.trace.append(f'unit found: c{i}') # : {log_clause(clause)}') # TODO convert clause
                    #self.trace.append('UP end')
                    propagated = False
                    return clause
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    #self.trace.append(f'variable assigned: x{var} = {value}') #TODO
                    self.assign(var, value, clause)
                    

            if not propagated:
                self.trace.append("nothing propagated")
                break
        self.trace.append("UP end")
        return None


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
num_vars = 7
formulas = []
for i in tqdm(range(100000)):
    formulas.append(generate_random_formula(num_vars))

traces =  []

for clauses in formulas:
    solver = CDCLSolver(clauses)
    # Sample random integer n from 3 to 5
    
    n = random.randint(2, num_vars-1)
    # Choose n random variables and assign random boolean values
    vars_to_assign = random.sample(range(1, num_vars+1), n)  # Choose n random variables from 1-10
    for var in vars_to_assign:
        value = random.choice([True, False])
        solver.assignments[var] = value
    solver.unit_propagate()
    traces.append(" ; ".join(solver.trace))

train_size = int(0.8 * len(traces))
train_traces = traces[:train_size]
test_traces = traces[train_size:]

# Create dictionaries with 'text' key
train_data = [{"text": trace} for trace in train_traces]
test_data = [{"text": trace} for trace in test_traces]

# Save to JSON files
if not os.path.exists("temp"):
    os.makedirs("temp")

with open("temp/cdcl_train.json", "w") as f:
    json.dump(train_data, f, indent=2)
    
with open("temp/cdcl_test.json", "w") as f:
    json.dump(test_data, f, indent=2)


