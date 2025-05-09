import sys
import os
import argparse
import json
import random
from tqdm import tqdm

from pathlib import Path
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cdcl.cdcl_solver import CDCLSolver
from src.cdcl.tracer import Tracer
from src.cdcl.data_gen_utils import generate_random_formula, remap_full_trace

parser = argparse.ArgumentParser(description="Generate SAT CDCL formula solver traces.")
parser.add_argument("--num_formulas", type=int, default=100000, help="Number of formulas to generate traces from.")
parser.add_argument("--n_vars_range", type=int, nargs=2, default=[3, 6], help="Range for number of variables (min max).")
parser.add_argument("--variance", type=float, default=0.1, help="Variance in number of clauses per formula.")
parser.add_argument(
    "--remap_vars_up_to", type=int, required=False,
    help="Enable variable remapping. Provide a number to override. 0 means no remap."
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--split", type=str, help="Which data file to use. Options: train, val, test, ood.")

cli_args, unknown = parser.parse_known_args()  # `unknown` gets passed to Hydra
sys.argv = [sys.argv[0]] + unknown  # Hydra now sees only unknowns or config overrides


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    random.seed(cli_args.seed)

    if cli_args.split is not None:
        split_cfg = cfg.data_generation.defaults[cli_args.split]

        # Override CLI args only if they were not set explicitly
        if cli_args.remap_vars_up_to == parser.get_default("remap_vars_up_to"):
            cli_args.remap_vars_up_to = cfg.data_generation.defaults.remap_vars_up_to
        if cli_args.num_formulas == parser.get_default("num_formulas"):
            cli_args.num_formulas = split_cfg.num_formulas
        if cli_args.n_vars_range == parser.get_default("n_vars_range"):
            cli_args.n_vars_range = split_cfg.n_vars_range
        if cli_args.variance == parser.get_default("variance"):
            cli_args.variance = split_cfg.variance
        if cli_args.seed == parser.get_default("seed"):
            cli_args.seed = split_cfg.seed

    print(
        f"[Config] Split: {cli_args.split}, "
        f"Formulas: {cli_args.num_formulas}, Vars: {cli_args.n_vars_range}, Seed: {cli_args.seed}"
    )

    formulas = []
    for _ in tqdm(range(cli_args.num_formulas), desc="Generating formulas..."):
        n_vars = int(random.uniform(cli_args.n_vars_range[0], cli_args.n_vars_range[1]))
        formulas.append((n_vars, generate_random_formula(n_vars=n_vars, variance=cli_args.variance)))

    traces = []
    n_sat = 0
    for n_vars, clauses in tqdm(formulas, desc="Creating the traces..."):
        solver = CDCLSolver(clauses, tracer=Tracer())
        is_satisfiable = solver.solve()
        n_sat += is_satisfiable

        trace = solver.tracer.get_trace()
        if cli_args.remap_vars_up_to != 0:
            trace = remap_full_trace(trace, n_vars, remap_up_to=cli_args.remap_vars_up_to, method='shift')
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

    data_path = Path(to_absolute_path(cfg.data.raw_files[cli_args.split]))
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the dataset to JSON file
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()

