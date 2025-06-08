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


def log_int(i):
    return ' '.join(str(i))

class CDCLSolver:
    def __init__(self, clauses,vars):
        self.clauses = clauses
        self.vars = vars
        self.clause2id = {tuple(c):f'c {log_int(i)}' for i,c in enumerate(clauses)}
        self.assignments = {}
        self.level = 0
        self.decision_level = {}
        self.implication_graph = {}
        self.reason_clauses = {}
        self.learned_clauses = []
        self.trace = []
        self.ret_dic = {"input_clauses": self.clause2id, 
                   "solve_traces": [],
                   "unit_prop_traces": [],
                   "analyze_conflict_traces": [],}
        
    def solve(self):
        while True:
            trace = [f"SOLVE_BEGIN"]
            trace.append(f"\nCALL_UNIT_PROPAGATION")
            conflict = self.unit_propagate()
            trace.append(f"\nREAD_ASSIGNMENTS READ_BEGIN {log_assignments(self.assignments)} READ_END")
            trace.append(f"\nREAD_CLAUSES READ_BEGIN {log_clause_list(self.clauses + self.learned_clauses,self.clause2id)} READ_END")
            trace.append(f"\nREAD_DECISION_LEVELS READ_BEGIN {log_decision_level(self.decision_level)} READ_END")
            trace.append(f"\nREAD_LEVEL READ_BEGIN {log_int(self.level)} READ_END")
            if conflict:
                trace.append(f"\nREAD_CONFLICT_CLAUSE READ_BEGIN {log_clause(conflict,self.clause2id[tuple(conflict)])} READ_END")
                trace.append("SPLIT_BEGIN")
                if self.level == 0:
                    trace.extend(["UNSAT", "\nSOLVE_END"])
                    self.ret_dic["solve_traces"].append(trace)
                    return False
                trace.append(f"\nCALL_ANALYZE_CONFLICT")
                learned_clause = self.analyze_conflict(conflict)
                trace.append(f"\nREAD_LEARNED_CLAUSE READ_BEGIN {log_new_clause(learned_clause)} READ_END")
                backtrack_level,trace_backtrack = self.find_backtrack_level(learned_clause)
                trace.extend(trace_backtrack)
                #trace.append(f"\nREAD_BACKTRACK_LEVEL {backtrack_level} READ END")
                self.backtrack(backtrack_level)
                self.learned_clauses.append(learned_clause)
                self.clause2id[tuple(learned_clause)] = f"c {' '.join(str(len(self.clause2id)))}"
            else:
                trace.append(f"\nCONFLICT 0")
                trace.append("SPLIT_BEGIN")
                trace.append(f'\nN_VARS_TOTAL {len(self.assignments)}')
                if len(self.assignments) == self.count_variables():
                    trace.extend(["SAT", "\nSOLVE_END"])
                    self.ret_dic["solve_traces"].append(trace)
                    return True
                var = self.pick_branching_variable()
                trace.append(f"\nBRANCHING_VARIABLE {log_var(var)}")
                if var is None:
                    return True
                self.level += 1
                self.assign(var, True, None)
                trace.append(f"\nWRITE_ASSIGNMENTS WRITE_BEGIN x{var} = {True} WRITE_END")
                trace.append(f"\nWRITE_DECISION_LEVELS WRITE_BEGIN {self.level} WRITE_END")
            trace.append("END ")
            self.ret_dic["solve_traces"].append(trace)

    def unit_propagate(self,):
        trace = []
        trace.append("UNIT_PROPAGATION_BEGIN")
        trace.append(f"\nREAD_ASSIGNMENTS READ_BEGIN {log_assignments(self.assignments)} READ_END")
        trace.append(f"\nREAD_CLAUSES READ_BEGIN {log_clause_list(self.clauses + self.learned_clauses, self.clause2id)} READ_END")
        trace.append(f"\nUP_BEGIN")
        while True:
            trace.append('SPLIT_BEGIN')
            propagated = False
            for i,clause in enumerate(self.clauses + self.learned_clauses):
                status, value = self.evaluate_clause(clause)
                trace.append(f"EVALUATE_CLAUSE {log_clause(clause,self.clause2id[tuple(clause)])}")
                if status and not value:
                    trace.append([
                        f"\nWRITE_CONFLICT_CLAUSE WRITE_BEGIN c {' '.join(str(i))} WRITE_END",
                        f"\nUNIT_PROPAGATION_END",
                    ])
                    self.ret_dic["unit_prop_traces"].append(trace)
                    propagated = False
                    return clause
                elif self.is_unit(clause):
                    trace.append(f'PROPAGATED') # : {log_clause(clause)}') # TODO convert clause
                    #self.trace.append('UP end')
                    lit = self.get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    propagated = True
                    #return clause
                    trace.append(f'\nWRITE_ASSIGNMENTS WRITE_BEGIN x{var} = {value} WRITE_END') #TODO
                    self.assign(var, value, clause)
                    trace.append(f"\nWRITE_DECISION_LEVELS WRITE_BEGIN {self.level} WRITE_END")
                    trace.append(f"\nWRITE_REASON_CLAUSES WRITE_BEGIN c {' '.join(str(i))} WRITE_END")
                    
            if not propagated:
                trace.append("\nNOTHING_PROPAGATED")
                break
        trace.append("END")
        self.ret_dic["unit_prop_traces"].append(trace)
        return None


    def analyze_conflict(self, conflict_clause):
        trace = []
        trace.append("ANALYZE_CONFLICT_BEGIN")
        trace.append(f"\nREAD_ASSIGNMENTS READ_BEGIN {log_assignments(self.assignments)} READ_END")
        trace.append(f"\nREAD_CLAUSES READ_BEGIN {log_clause_list(self.clauses + self.learned_clauses,self.clause2id)} READ_END")
        trace.append(f"\nREAD_DECISION_LEVELS READ_BEGIN {log_decision_level(self.decision_level)} READ_END")
        trace.append(f"\nREAD_REASON_CLAUSES READ_BEGIN {log_reasons(self.reason_clauses,self.clause2id)} READ_END")
        trace.append(f"\nREAD_CONFLICT_CLAUSE READ_BEGIN {log_clause(conflict_clause,self.clause2id[tuple(conflict_clause)])} READ_END")
        trace.append("SPLIT_BEGIN")
        # Initialize sets to track variables at current decision level and literals for learned clause
        current_level_vars = set()  # Variables assigned at current decision level
        learned_lits = set()        # Literals that will form the learned clause

        # Start with literals from the conflict clause
        queue = self.get_literals_from_clause(conflict_clause) #
        trace.append(f"\QUEUE {log_queue(queue)}")
        while True:
            # Process each literal in the current clause
            for lit in queue:
                var = abs(lit)  # Get variable (removing sign)
                trace.append(f"\nCHECKING_VARIABLE {log_var(var)} AT_LEVEL {self.decision_level.get(var)}")
                
                # If variable was assigned at current level, add to current_level_vars
                if self.decision_level.get(var) == self.level:
                    current_level_vars.add(var)
                    trace.append(f"\nCURRENT_LEVEL_VARS {log_current_level_vars(current_level_vars)}") #TODO
                # If assigned at earlier level, add to learned clause
                else:
                    learned_lits.add(-var if self.assignments[var] else var)
                    var_str = log_var(var)
                    trace.append(f"\nLEARNED_LITS { '- ' + var_str if self.assignments[var] else '+ ' + var_str}") #TODO
            
            # UIP condition: only one variable from current decision level remains
            if len(current_level_vars) <= 1:
                trace.append("\nUIP")
                break

                
            # Get most recently assigned variable from current level
            var = self.get_latest_assigned(current_level_vars)
            trace.append(f"\nLATEST_ASSIGNED: {log_var(var)}")
            current_level_vars.remove(var)
            
            # Get the clause that caused this variable's assignment
            reason = self.reason_clauses.get(var)
            trace.append(f"\nREASON_FOR {log_var(var)} IS {self.clause2id[tuple(reason)]}") #TODO
            if reason:
                queue = [lit for lit in self.get_literals_from_clause(reason) 
                        if abs(lit) != var]
                trace.append(f"\nQUEUE {log_queue(queue)}") # TODO
        
        # Create set of literals from current level variables with opposite polarity
        current_level_lits = [-var if self.assignments[var] else var for var in current_level_vars]
        trace.append(f"\nCURRENT_LEVEL_LITS: {log_queue(current_level_lits)}")
        new_clause =list(learned_lits.union(set(current_level_lits))) # TODO check if this is correct'
        # trace.append(f"new-clause: {log_new_clause(new_clause)}")
        trace.append(f"\nWRITE_LEARNED_CLAUSES WRITE_BEGIN {log_new_clause(new_clause)} WRITE_END")
        trace.append("END")
        # self.trace.append(trace)
        self.ret_dic["analyze_conflict_traces"].append(trace)
        
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

    def log_levels(self, levels):
        string = '{ '
        for level in levels:
            string += f"{level} , "
        return string[:-3] + ' }'
    
    def find_backtrack_level(self, learned_clause):
        trace = []
        trace.append(f"FB_BEGIN")
        levels = [self.decision_level[abs(lit)] for lit in learned_clause 
                 if abs(lit) in self.decision_level]
        trace.append(f"LEVELS {self.log_levels(levels)}")
        if not levels:
            trace.append("GOTO_LEVEL 0")
            return 0, trace
        levels.sort(reverse=True)
        goto = levels[1] if len(levels) > 1 else 0
        trace.append(f"GOTO_LEVEL {goto}")
        return goto, trace

    def pick_branching_variable(self):
        for var in self.vars:
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
    interval_start = random.randint(1, 26 - n_vars)
    var_range = range(interval_start, interval_start + n_vars + 1)
    clauses = []
    for _ in range(n_clauses):
        clause_vars = random.sample(var_range, clause_length)
        clause = [var if random.random() < 0.5 else -var for var in clause_vars]
        clauses.append(clause)
    return (clauses, var_range)

######## TEST
formulas = []
for i in tqdm(range(100)):
    num_vars = random.randint(5, 15)
    formulas.append(generate_random_formula(num_vars))

# for i in tqdm(range(5000)):
#     num_vars = 25
#     formulas.append(generate_random_formula(num_vars))

# for i in tqdm(range(2000)):
#     num_vars = random.randint(16, 25)
#     formulas.append(generate_random_formula(num_vars))

traces = []
for clauses, vars in tqdm(formulas):
    g = Glucose3()
    for clause in clauses:
        g.add_clause(clause)
    is_sat_pysat = g.solve()
    g.delete()

    solver = CDCLSolver(clauses, vars)
    is_satisfiable = solver.solve()
    traces.append(solver.ret_dic)
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

# Process traces to match desired output structure
import json
trace_strs = []
for trace in traces:
    # Convert input_clauses to string format
    input_clauses_str = ""
    if isinstance(trace["input_clauses"], dict):
        # Convert clause2id dict to string representation
        clauses_list = []
        for clause_tuple, clause_id in trace["input_clauses"].items():
            clause = list(clause_tuple)
            clause_str = "[ "
            for lit in clause:
                if lit > 0:
                    clause_str += f"+ x{abs(lit)} "
                else:
                    clause_str += f"- x{abs(lit)} "
            clause_str += "]"
            clauses_list.append(clause_str)
        input_clauses_str = " ".join(clauses_list)
    else:
        input_clauses_str = str(trace["input_clauses"])
    
    # Concatenate all solve traces into one string (preserving newlines)
    solve_trace_str = []
    for solve_trace in trace["solve_traces"]:
        if isinstance(solve_trace, list):
            # Join list elements, preserving any \n characters within strings
            solve_trace_str.append(" ; ".join(str(item) for item in solve_trace))
        else:
            solve_trace_str.append(str(solve_trace))
    
    # Convert unit prop traces to list of strings (preserving newlines)
    unit_prop_traces_strs = []
    for unit_trace in trace["unit_prop_traces"]:
        if isinstance(unit_trace, list):
            # Join list elements, preserving any \n characters within strings
            unit_prop_traces_strs.append(" ; ".join(str(item) for item in unit_trace))
        else:
            unit_prop_traces_strs.append(str(unit_trace))
    
    # Convert analyze conflict traces to list of strings (preserving newlines)
    analyze_conflict_traces_strs = []
    for conflict_trace in trace["analyze_conflict_traces"]:
        if isinstance(conflict_trace, list):
            # Join list elements, preserving any \n characters within strings
            analyze_conflict_traces_strs.append(" ; ".join(str(item) for item in conflict_trace))
        else:
            analyze_conflict_traces_strs.append(str(conflict_trace))
    
    # Create the final structure as a dictionary (don't convert to JSON string yet)
    formatted_trace = {
        "input_clauses": input_clauses_str,
        "solve_trace": solve_trace_str,
        "unit_prop_traces": unit_prop_traces_strs,
        "analyze_conflict_traces": analyze_conflict_traces_strs
    }
    
    trace_strs.append(formatted_trace)

# Now trace_strs[0] will have the desired structure
print(json.dumps(trace_strs[0], indent=2))

num_traces = len(trace_strs)
split_idx = int(0.90 * num_traces)

train_traces = trace_strs[:split_idx]
test_traces = trace_strs[split_idx:]

print(f"Split {num_traces} traces into {len(train_traces)} training and {len(test_traces)} testing traces")

# Save the splits to files


# Save traces directly as JSON objects (no "text" wrapper needed)
train_data = train_traces
test_data = test_traces

# # Save to JSON files
# with open('train_CA.json', 'w') as f:
#     json.dump(train_data, f, indent=2)

with open('test_CA.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Saved training traces to train_traces.json")
print(f"Saved testing traces to test_traces.json")