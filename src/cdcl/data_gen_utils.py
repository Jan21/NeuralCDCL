import random
import re
from typing import Optional


def remap_full_trace(
    full_trace: dict[str, list[str] | list[list[str]]],
    n_vars: int,
    remap_up_to: int,
    method: str = "random",   # "random" or "shift"
) -> dict[str, list[str] | list[list[str]]]:
    """
    Remaps all variable names in the trace using the specified method.

    Args:
        full_trace: Dictionary of trace sections, each a list of logs or list of list of logs.
        n_vars: Number of original variables.
        remap_up_to: Used in "random" mode, max variable index allowed.
        method: "random" for random remap, "shift" for index shifting.

    Returns:
        Remapped trace with updated variable indices.
    """
    if method == "random":
        new_indices = sorted(random.sample(range(1, remap_up_to + 1), n_vars))
        mapping = {i: new_indices[i - 1] for i in range(1, n_vars + 1)}
        remapper = lambda s: remap_trace_variables(s, n_vars, mapping)
    elif method == "shift":
        max_valid_shift = remap_up_to - n_vars
        shift = random.randint(1, max_valid_shift)
        remapper = lambda s: remap_trace_variables_shifted(s, n_vars, remap_up_to, shift)
    else:
        raise ValueError(f"Unknown remap method: {method}")

    return {
        name: [
            [remapper(log) for log in logs] if isinstance(logs, list)
            else remapper(logs)
            for logs in spec_trace
        ]
        for name, spec_trace in full_trace.items()
    }


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


def remap_trace_variables_shifted(trace_text: str, n: int, remap_up_to: int, shift: int) -> str:
    """
    Remaps variables x1..xn to x{i+shift}, ensuring new indices do not exceed remap_up_to.

    Args:
        trace_text (str): Input trace string with variables like x1, x2, etc.
        n (int): Number of original variables (x1 to xn).
        remap_up_to (int): Maximum allowed index after shift.
        shift (int): Shift amount to apply to variable indices.

    Returns:
        str: Trace with remapped variable indices.

    Raises:
        ValueError: If any shifted variable index exceeds remap_up_to.
    """
    pattern = re.compile(r"(-?)x(\d+)")

    def replace_var(match):
        sign = match.group(1)
        old_index = int(match.group(2))

        if not (1 <= old_index <= n):
            return match.group(0)

        new_index = old_index + shift
        if new_index > remap_up_to:
            raise ValueError(f"Shifted variable x{old_index} -> x{new_index} exceeds remap_up_to={remap_up_to}")

        return f"{sign}x{new_index}"

    return pattern.sub(replace_var, trace_text)


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