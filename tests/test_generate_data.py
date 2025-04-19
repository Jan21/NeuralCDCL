from src.cdcl.data_gen_utils import generate_random_formula
from src.cdcl.cdcl_solver import CDCLSolver
from src.cdcl.tracer import Tracer
from pysat.solvers import Glucose3
import random


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


def test_cdcl_solver():
    num_formulas = 20
    n_vars_range = (5, 20)
    n_clauses_var = 0.5

    formulas = []
    for _ in range(num_formulas):
        n_vars = int(random.uniform(n_vars_range[0], n_vars_range[1]))
        formulas.append((n_vars, generate_random_formula(n_vars=n_vars, variance=n_clauses_var)))

    n_sat = 0
    for n_vars, clauses in formulas:
        solver = CDCLSolver(clauses, tracer=Tracer())
        is_satisfiable = solver.solve()
        n_sat += is_satisfiable

        validate_cdcl_with_pysat(clauses, is_satisfiable, solver.assignments)