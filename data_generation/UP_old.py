from pysat.solvers import Glucose3
import random
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
import os
import argparse

# def log_clauses(clauses):
#     string = 'clauses: '
#     for c in clauses:
#         string += "( "
#         for l in c:
#             if l > 0:
#                 string += f"+ {abs(l)} "
#             else:
#                 string += f"- {abs(l)} "
#         string += ") "
#     string = string.strip()
#     return string

# def log_clauses(clauses):
    
#     litdic = defaultdict(list)
#     clausedic = defaultdict(list)
#     for i,cl in enumerate(clauses):
#         for l in cl:
#             litdic[l].append(i+1)
#             clausedic[i+1].append(l)
    
#     token_positions = []
    
#     # Add litdic lists
#     for lit, clauses in litdic.items():
#         token_positions.append(clauses)
        
#     # Add clausedic lists 
#     for clause_num, lits in clausedic.items():
#         token_positions.append(lits)
#     # Find maximum length of lists in result
#     max_length = max(len(lst) for lst in token_positions)
#     lit_dic_np = {}
#     for k,v in litdic.items():
#         padded_v = v + [0] * (max_length - len(v))
#         lit_dic_np[k] = np.array(padded_v)

#     clause_dic_np = {}
#     for k,v in clausedic.items():
#         padded_v = v + [0] * (max_length - len(v))
#         clause_dic_np[k] = np.array(padded_v)

        
#     return (clause_dic_np, lit_dic_np)

def log_clause(clause, i):
    string = f'( '
    for l in clause:
        abs_l = abs(l)
        abs_l_str = ' '.join(digit for digit in str(abs_l))
        
        if l > 0:
            string += f"+ x {abs_l_str} "
        else:
            string += f"- x {abs_l_str} "
    
    i_str = ' '.join(digit for digit in str(i))
    
    return string.strip() + f' ) : c {i_str}'

def log_clause_list(clause_list):
    string = ''
    for i,clause in enumerate(clause_list):
        string += log_clause(clause,i) + ' , '
    return string[:-3]

def log_assignments(assignments):
    sorted_assignments = sorted(assignments.items(), key=lambda x: x[0])
    assignments_str = ""
    for var, value in sorted_assignments:
        var_str = ' '.join(digit for digit in str(var))
        assignments_str += f"x {var_str} = {value} , "
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
                    i_str = ' '.join(digit for digit in str(i))
                    self.trace.append(f"found conflict: c {i_str}") #{log_clause(clause,i)}") #TODO
                    self.trace.append('UP end')
                    propagated = False
                    return clause
                elif self.is_unit(clause):
                    i_str = ' '.join(digit for digit in str(i))
                    self.trace.append(f'unit found: c {i_str}') # : {log_clause(clause)}') # TODO convert clause
                    #self.trace.append('UP end')
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    propagated = True
                    #return clause
                    var_str = ' '.join(digit for digit in str(var))
                    self.trace.append(f'variable assigned: x {var_str} = {value}') #TODO
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


def generate_random_formula(n_vars, coefficient=4.2, clause_length=3):
    """
    Generate a random SAT formula with n variables.
    Args:
        n_vars: Number of variables
        n_clauses: Number of clauses (default: around 4.2 * n_vars for balanced SAT/UNSAT)
        coefficient: Coefficient for determining number of clauses (default: 4.2)
        clause_length: Length of each clause (default: 3 for 3-SAT)
    Returns:
        List of clauses, where each clause is a list of integers
    """
    # Use empirically determined ratio for balanced SAT/UNSAT
    n_clauses = int(coefficient * n_vars)
    # vygenerovat nebo jako argument set proměnných které mohu použít
    # z nich na 193 budu samplovat
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

parser = argparse.ArgumentParser(description='Generate and test random SAT formulas')
parser.add_argument('--coefficient', type=float, default=4.2,
                    help='Coefficient for determining number of clauses (default: 4.2)')
parser.add_argument('--num_vars', type=int, default=7,
                    help='Number of variables in the formulas (default: 7)')
args = parser.parse_args()

coefficient = args.coefficient

 ######## TEST
num_vars = args.num_vars
formulas = []
for i in tqdm(range(200128)):
    formulas.append(generate_random_formula(num_vars, coefficient))

traces =  []

for clauses in tqdm(formulas):
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

# train_size = int(0.9 * len(traces))
# train_traces = traces[:train_size]
# test_traces = traces[train_size:]

# všechny možné dvojciferné čísla, odebrat ty které budou v test setu
# do test setu  1 1 2 2 3 3 4 4
# jen 2 číselné proměnné
# první vygenerovat formule, pak vytáhnout čísla které se v každé použily
# z těch které se použily se na 227 vybere n

# for generalization:
train_size = 200000
train_traces = traces[:train_size]
test_traces = traces[train_size:train_size+128]

# Create dictionaries with 'text' key
train_data = [{"text": trace} for trace in train_traces]
test_data = [{"text": trace} for trace in test_traces]

# Save to JSON files
if not os.path.exists("temp"):
    os.makedirs("temp")
if not os.path.exists("temp/train"):
    os.makedirs("temp/train")
    os.makedirs("temp/test")
if not os.path.exists("temp/generalization"):
    os.makedirs("temp/generalization")

# Update the filenames to include parameters
train_filename = f"temp/train/cdcl_train_vars{num_vars}_coef_{coefficient}.json"
test_filename = f"temp/test/cdcl_test_vars{num_vars}_coef_{coefficient}.json"
generalization_test_filename = f"temp/generalization/cdcl_test_vars{num_vars}_coef_{coefficient}.json"
# Use these filenames instead of the original one

with open(train_filename, "w") as f:
    json.dump(train_data, f, indent=2)
    
with open(test_filename, "w") as f:
    json.dump(test_data, f, indent=2)

with open(generalization_test_filename, "w") as f:
    json.dump(test_data, f, indent=2)


