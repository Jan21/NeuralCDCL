import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import random
import re
from typing import Optional
from pysat.solvers import Glucose3

from tqdm import tqdm
from src.cdcl.cdcl import CDCLSolver
from src.cdcl.tracer import Tracer


def validate_cdcl_with_pysat(clauses: list[list[int]], is_satisfiable: bool, assignments: dict[int, bool]) -> None:
    """
    Validates the output of the CDCL solver using PySAT.
    Raises an assertion error if the validation fails.
    """
    pysat_solver = Glucose3()
    for clause in clauses:
        pysat_solver.add_clause(clause)

    # 1. Check that PySAT agrees with CDCL on satisfiability
    assert pysat_solver.solve() == is_satisfiable, "Mismatch with PySAT result"

    # 2. If satisfiable, check that the CDCL assignment satisfies the formula
    if is_satisfiable:
        assumptions = [var if val else -var for var, val in assignments.items()]
        assert pysat_solver.solve(assumptions=assumptions), "CDCL assignment does not satisfy formula"

    pysat_solver.delete()

def remap_trace_variables(trace_text: str, n: int, mapping: dict[int, int]) -> str:
    """
    Remaps variables x1..x_n to x_{f(1)}..x_{f(n)} in a monotonic way,
    where f is a strictly increasing mapping from {1..n} to {1..N}.
    """
    # 2) Regex to match a variable possibly with a leading minus:
    #    We look for an optional minus sign, then "x", then one or more digits.
    #    We also ensure we only remap if the digits are within 1..n.
    pattern = re.compile(r"(-?)x(\d+)")
    
    def replace_var(match):
        sign      = match.group(1)       # The optional '-' sign
        old_index = int(match.group(2))  # The integer part after 'x'
        
        # If it's outside 1..n, leave it unchanged (or handle error if appropriate)
        if not (1 <= old_index <= n):
            return match.group(0)  # return the original match with no change
        
        # Otherwise, map to x_{f(old_index)}
        new_index = mapping[old_index]
        return f"{sign}x{new_index}"
    
    # 3) Run the substitution
    remapped_text = pattern.sub(replace_var, trace_text)
    return remapped_text

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

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    formulas = []
    for _ in tqdm(range(args.num_formulas), desc="Generating formulas..."):
        n_vars = int(random.uniform(args.n_vars_range[0], args.n_vars_range[1]))
        formulas.append((n_vars, generate_random_formula(n_vars=n_vars, variance=args.variance)))

    traces = []
    n_sat = 0
    for n_vars, clauses in tqdm(formulas, desc="Creating the traces..."):
        solver = CDCLSolver(clauses, tracer=Tracer())
        is_satisfiable = solver.solve()
        n_sat += is_satisfiable

        if args.validate_with_pysat:
            validate_cdcl_with_pysat(clauses, is_satisfiable, solver.assignments)

        trace = solver.tracer.get_trace()
        if args.remap_variables != 0:
            new_indices = sorted(random.sample(range(1, args.remap_variables + 1), n_vars))
            mapping = { i: new_indices[i-1] for i in range(1, n_vars+1) }
            trace = {
                name: [
                    [
                        remap_trace_variables(log, n_vars, mapping) for log in logs
                    ] if isinstance(logs, list)
                    else remap_trace_variables(logs, n_vars, mapping)
                    for logs in spec_trace
                ]
                for name, spec_trace in trace.items()
            }
        traces.append(trace)
    sat_ratio = n_sat / len(formulas)
    print(f'SAT/UNSAT ratio = {sat_ratio}')

    # Prepare data for output
    data = []
    for trace in tqdm(traces, desc="Writing to file..."):
        data_entry = {
            "input_clauses": '\n'.join(trace['input_clauses']),
            "solve_trace": '\n'.join(trace['solve_trace']),
            "unit_prop_traces": ['\n'.join(logs) for logs in trace['unit_prop_traces']],
            "analyze_conflict_traces": ['\n'.join(logs) for logs in trace['analyze_conflict_traces']],
        }
        data.append(data_entry)

    # Write the dataset to JSON file
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAT CDCL formula solver traces.")
    parser.add_argument("--num_formulas", type=int, default=100000, help="Number of formulas to generate traces from.")
    parser.add_argument("--n_vars_range", type=int, nargs=2, default=[3, 6], help="Range for number of variables (min max).")
    parser.add_argument("--variance", type=float, default=0.1, help="Variance in number of clauses per formula.")
    parser.add_argument(
        "--remap_variables", nargs="?", const=25, type=int, default=0,
        help="Enable variable remapping. Provide a number to override (e.g., --remap_variables 6)."
    )
    parser.add_argument("--output_file", type=str, default="cdcl_dataset.json", help="Output file to store the results.")
    parser.add_argument(
        "--validate_with_pysat",
        action="store_true",
        help="Validate the CDCL result using PySAT for correctness."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args)

