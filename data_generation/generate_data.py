import re
import random
import argparse
import json
from tqdm import tqdm
from CDCL import CDCLSolver
from tracer import Tracer, format_list

def remap_variables(trace_text, n, mapping):
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

def generate_random_formula(n_vars, n_clauses = None, clause_length=3):
    balanced_n_clauses = {3: 19, 4: 24, 5: 28, 6: 33, 7: 37, 8: 41, 9: 45, 10: 50, 11: 54, 12: 58, 
                          13: 63, 14: 67, 15: 71, 16: 76, 17: 79, 18: 83, 19: 87, 20: 92}
    if n_clauses == None:
        n_clauses = balanced_n_clauses[n_vars] if n_vars in balanced_n_clauses else n_vars * 4.26
    var_range = range(1, n_vars + 1)
    clauses = []
    for _ in range(n_clauses):
        clause_vars = random.sample(var_range, clause_length)
        clause = [var if random.random() < 0.5 else -var for var in clause_vars]
        clauses.append(clause)
    return clauses

def main(args):
    formulas = []
    for _ in tqdm(range(args.num_formulas), desc="Generating formulas..."):
        n_vars = int(random.uniform(args.n_vars_range[0], args.n_vars_range[1]))
        formulas.append((n_vars, generate_random_formula(n_vars=n_vars)))

    all_logs = []
    n_sat = 0
    for n_vars, clauses in tqdm(formulas, desc="Creating the traces..."):
        solver = CDCLSolver(clauses, tracer=Tracer())
        is_satisfiable = solver.solve()
        n_sat += is_satisfiable
        tracer = solver.get_tracer()
        packed_logs = tracer.get_trace(packed=True)
        unpacked_logs = tracer.get_trace(packed=False)
        if remap_variables:
            new_indices = sorted(random.sample(range(1, args.remap_variables + 1), n_vars))
            mapping = { i: new_indices[i-1] for i in range(1, n_vars+1) }
            packed_logs = {
                name: [remap_variables(' '.join(trc), n_vars, mapping) for trc in traces]
                for name, traces in packed_logs.items()
            }
            unpacked_logs = remap_variables(' '.join(unpacked_logs), n_vars, mapping)
        all_logs.append((packed_logs, unpacked_logs))
    sat_ratio = n_sat / len(formulas)
    print(f'SAT/UNSAT ratio = {sat_ratio}')

    # Prepare data for output
    data = []
    for (n_vars, clauses), (packed_logs, unpacked_logs) in tqdm(zip(formulas, all_logs), desc="Writing to file..."):
        data_entry = {
            "formula": format_list(clauses, is_var=True),
            "solve_trace_packed": packed_logs['solve_traces'][0],
            "solve_trace_unpacked": unpacked_logs,
        }
        data_entry["unit_prop_traces"] = [trace for trace in packed_logs['unit_prop_traces']]
        data_entry["analyze_conflict_traces"] = [trace for trace in packed_logs['analyze_conflict_traces']]
        data.append(data_entry)

    # Write the dataset to JSON file
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAT CDCL formula solver traces.")
    parser.add_argument("--num_formulas", type=int, default=100000, help="Number of formulas to generate traces from.")
    parser.add_argument("--n_vars_range", type=int, nargs=2, default=[3, 6], help="Range for number of variables (min max).")
    parser.add_argument(
        "--remap_variables", nargs="?", const=25, type=int, default=10,
        help="Enable variable remapping. If enabled without a value, defaults to N=10. Provide a number to override (e.g., --remap_variables 6)."
    )
    parser.add_argument("--output_file", type=str, default="cdcl_dataset.json", help="Output file to store the results.")
    args = parser.parse_args()
    main(args)

