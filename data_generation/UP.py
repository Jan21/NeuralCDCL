import random
import json
import os
import argparse
from tqdm import tqdm

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
                    self.trace.append(f"found conflict: c {i_str}")
                    self.trace.append('UP end')
                    propagated = False
                    return clause
                elif self.is_unit(clause):
                    i_str = ' '.join(digit for digit in str(i))
                    self.trace.append(f'unit found: c {i_str}')
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    propagated = True
                    var_str = ' '.join(digit for digit in str(var))
                    self.trace.append(f'variable assigned: x {var_str} = {value}')
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

def generate_random_formula(n_vars, allowed_vars, coefficient=4.2, clause_length=3):
    """
    Generate a random SAT formula using only variables from allowed_vars.
    """
    n_clauses = int(coefficient * len(allowed_vars))
    
    # Convert allowed_vars to a list for random sampling
    vars_list = list(allowed_vars)
    
    clauses = []
    
    for _ in range(n_clauses):
        # Generate a clause with random literals
        clause = []
        clause_vars = set()
        
        while len(clause) < clause_length:
            # If we've used all available variables for this clause, break
            if len(clause_vars) >= len(vars_list):
                break
                
            # Pick a random variable from allowed variables that hasn't been used in this clause
            var = random.choice(vars_list)
            if var not in clause_vars:
                # Randomly choose positive or negative literal
                lit = var if random.random() < 0.5 else -var
                clause.append(lit)
                clause_vars.add(var)
                
        if clause:  # Only add non-empty clauses
            clauses.append(clause)
        
    return clauses

def main():
    parser = argparse.ArgumentParser(description='Generate and test random SAT formulas')
    parser.add_argument('--coefficient', type=float, default=4.2,
                        help='Coefficient for determining number of clauses (default: 4.2)')
    parser.add_argument('--num_vars', type=int, default=99,
                        help='Maximum variable number to consider (default: 99)')
    args = parser.parse_args()

    coefficient = args.coefficient
    max_var = args.num_vars
    
    # Define reserved variables for test set (variables where both digits are the same)
    reserved_test_vars = []
    for i in range(1, 10):  # For variables 11, 22, ..., 99
        double_digit = i * 10 + i
        if double_digit <= max_var:
            reserved_test_vars.append(double_digit)
    
    # Define available variables for training (all multi-digit variables except reserved test vars)
    train_vars = [i for i in range(10, max_var + 1) if i not in reserved_test_vars]  # Start from 10 to exclude single digits
    
    print(f"Reserved test-only variables: {reserved_test_vars}")
    print(f"Available training variables: {len(train_vars)} variables")
    
    # Generate train and test formulas
    train_size = 500000
    test_size = 500
    total_size = train_size + test_size
    
    formulas = []
    
    print("Generating formulas...")
    # Generate training formulas (no test-only variables)
    for i in tqdm(range(total_size)):
        if i < train_size:
            # For training: use only train_vars
            num_vars_to_use = min(random.randint(5, 20), len(train_vars))
            vars_for_formula = set(random.sample(train_vars, num_vars_to_use))
        else:
            # For testing: use any variables (both train and test vars)
            num_vars_to_use = min(random.randint(5, 9), len(reserved_test_vars))
            vars_for_formula = set(random.sample(reserved_test_vars, num_vars_to_use))
        
        formula = generate_random_formula(num_vars_to_use, vars_for_formula, coefficient)
        formulas.append(formula)
    
    print("Processing formulas to generate traces...")
    # Process all formulas to create traces
    traces = []
    for clauses in tqdm(formulas):
        solver = CDCLSolver(clauses)
        
        # Determine which variables are available for assignment
        used_vars = set()
        for clause in clauses:
            for lit in clause:
                used_vars.add(abs(lit))
        
        if len(used_vars) > 1:
            # Sample random integer n from 2 to the number of variables used
            n = random.randint(2, min(len(used_vars), len(used_vars) - 1))
            
            # Choose n random variables from used_vars
            vars_to_assign = random.sample(list(used_vars), n)
            
            for var in vars_to_assign:
                value = random.choice([True, False])
                solver.assignments[var] = value
        
        solver.unit_propagate()
        traces.append(" ; ".join(solver.trace))
    
    # Split into train and test sets
    train_traces = traces[:train_size]
    test_traces = traces[train_size:train_size+test_size]
    
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
    
    train_filename = f"temp/train/cdcl_train_vars{max_var}_coef_{coefficient}.json"
    test_filename = f"temp/test/cdcl_test_vars{max_var}_coef_{coefficient}.json"
    generalization_test_filename = f"temp/generalization/cdcl_test_vars{max_var}_coef_{coefficient}.json"
    
    with open(train_filename, "w") as f:
        json.dump(train_data, f, indent=2)
        
    with open(test_filename, "w") as f:
        json.dump(test_data, f, indent=2)
    
    with open(generalization_test_filename, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nFiles saved successfully:")
    print(f"  Training data: {train_filename}")
    print(f"  Test data: {test_filename}")
    print(f"  Generalization test data: {generalization_test_filename}")

if __name__ == "__main__":
    main()