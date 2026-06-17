from typing import Dict, List

import pandas as pd

from .core.budget import POI_DIMENSION_COLUMNS, generate_random_spatial_allocations
from .core.evaluation import (
    build_objective_state_nd,
    compute_baseline_iqc_total,
    get_available_dimensions,
    validate_hex_time_matrix,
)
from .core.io import load_seeds
from .core.results import persist_run
from .core.types import MetaheuristicContext
from .methods import METHOD_RUNNERS


METAHEURISTIC_METHODS = {
    'A': 'Iterated Local Search (ILS)',
    'B': 'Greedy Randomized Adaptive Search Procedure (GRASP)',
    'C': 'Biased Random-Key Genetic Algorithm (BRKGA)',
    'D': 'Particle Swarm Optimization (PSO)',
    'E': 'Hybrid Method (GRASP + VNS + Path Relinking)',
}

def ask_metaheuristic_method() -> str:
    """Prompt user to choose a metaheuristic method."""
    while True:
        print("\nSelect metaheuristic method:")
        for code, name in METAHEURISTIC_METHODS.items():
            print(f"{code}) {name}")
        choice = input("Enter method option [A-E]: ").strip().upper()
        if choice in METAHEURISTIC_METHODS:
            return choice
        print("Invalid option. Choose A, B, C, D, or E.")


def walk_meta_opt(df_walkability: pd.DataFrame,
                  df_hex_time_matrix: pd.DataFrame,
                  budget: int,
                  method: str,
                  seeds: List[int],
                  walking_profile: str,
                  location: str = None,
                  key_location: str = None,
                  h3_resolution: int = None,
                  distance: int = None) -> Dict[str, object]:
    """
    Prepare the metaheuristic optimization scenario from:
    - df_walkability (base indicators by hexagon)
    - df_hex_time_matrix (source-target temporal impact matrix)
    """
    if df_walkability is None or df_walkability.empty:
        raise ValueError("df_walkability is empty.")
    if df_hex_time_matrix is None or df_hex_time_matrix.empty:
        raise ValueError("df_hex_time_matrix is empty.")
    if budget <= 0:
        raise ValueError("BUDGET must be greater than zero.")
    if method not in METAHEURISTIC_METHODS:
        raise ValueError(f"Invalid method '{method}'.")
    if not seeds:
        raise ValueError("Seeds list is empty.")

    if 'h3_id' not in df_walkability.columns:
        raise ValueError("df_walkability is missing required column 'h3_id'.")

    df_walkability = df_walkability.copy()
    df_walkability['h3_id'] = df_walkability['h3_id'].astype(str)
    duplicated_h3 = int(df_walkability['h3_id'].duplicated().sum())
    if duplicated_h3 > 0:
        print(f"Warning: {duplicated_h3} duplicated h3_id rows found. Keeping first occurrence.")
        df_walkability = df_walkability.drop_duplicates(subset=['h3_id'], keep='first').reset_index(drop=True)

    df_hex_time_matrix = validate_hex_time_matrix(df_hex_time_matrix)

    dimensions = get_available_dimensions(df_walkability, POI_DIMENSION_COLUMNS)
    if not dimensions:
        raise ValueError("No POI dimension columns were found in df_walkability.")

    walkability_hex_ids = set(df_walkability['h3_id'].dropna().astype(str).tolist())
    matrix_source_hex_ids = set(df_hex_time_matrix['source_h3_id'].dropna().astype(str).tolist())
    source_hex_ids = sorted(walkability_hex_ids.intersection(matrix_source_hex_ids))
    if not source_hex_ids:
        raise ValueError("No common source hexagons between df_walkability and df_hex_time_matrix.")

    baseline_iqc_total = compute_baseline_iqc_total(df_walkability)
    objective_state_nd = build_objective_state_nd(
        df_walkability=df_walkability,
        df_hex_time_matrix=df_hex_time_matrix,
        candidate_dimensions=dimensions,
    )
    allocations = generate_random_spatial_allocations(
        budget=budget,
        dimensions=dimensions,
        source_hex_ids=source_hex_ids,
        seeds=seeds,
    )

    context = MetaheuristicContext(
        df_walkability=df_walkability,
        df_hex_time_matrix=df_hex_time_matrix,
        budget=int(budget),
        method_code=method,
        method_name=METAHEURISTIC_METHODS[method],
        seeds=[int(s) for s in seeds],
        walking_profile=walking_profile,
        dimensions=dimensions,
        source_hex_ids=source_hex_ids,
        baseline_iqc_total=baseline_iqc_total,
        allocations=allocations,
        objective_state_nd=objective_state_nd,
    )

    method_runner = METHOD_RUNNERS.get(method)
    method_result = method_runner(context) if method_runner else {
        'method_code': method,
        'method_name': METAHEURISTIC_METHODS[method],
        'status': 'missing_runner',
        'message': 'No method runner is registered for this method.',
    }

    persisted_paths = persist_run(
        context=context,
        method_result=method_result,
        location=location,
        key_location=key_location,
        h3_resolution=h3_resolution,
        distance=distance,
    )

    print("\nMetaheuristic setup initialized.")
    print(f"Method: {METAHEURISTIC_METHODS[method]} ({method})")
    print(f"Profile: {walking_profile}")
    print(f"BUDGET: {budget}")
    print(f"Seeds loaded: {len(seeds)}")
    print(f"POI dimensions: {', '.join(dimensions)}")
    print(f"Source hexagons available for allocation: {len(source_hex_ids)}")
    print(f"Hex-time matrix rows: {len(df_hex_time_matrix)}")
    if baseline_iqc_total is not None:
        print(f"Baseline IQC total: {baseline_iqc_total:.4f}")
    first_alloc = allocations[0]
    non_zero_count = len(first_alloc['allocation'])
    print(f"First seed allocation example ({first_alloc['seed']}): {non_zero_count} non-zero (hex,dimension) entries")
    print(f"Method module status: {method_result.get('status', 'unknown')}")

    return {
        'method_code': context.method_code,
        'method_name': context.method_name,
        'profile_key': context.walking_profile,
        'budget': context.budget,
        'seeds': context.seeds,
        'dimensions': context.dimensions,
        'source_hex_ids': context.source_hex_ids,
        'baseline_iqc_total': context.baseline_iqc_total,
        'allocations': allocations,
        'method_result': method_result,
        'persisted_files': persisted_paths,
    }
