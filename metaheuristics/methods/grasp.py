"""Greedy Randomized Adaptive Search Procedure (GRASP).

Implements the classic two-phase GRASP (Resende & Ribeiro):
    - greedy randomized construction (semi-greedy, RCL controlled by alpha),
    - first-improving local search,
    - multi-start: one construction + local search per seed, best solution kept.

Problem mapping (maximization of total IQC):
    - ground-set element: one POI unit assigned to a (source_hex, dimension) pair;
    - a solution is a candidate_matrix (n_hex x n_candidate_dimensions);
    - feasibility: exactly `budget` POIs allocated (construction stops at `budget`,
      local search only *moves* POIs, so no repair is ever needed);
    - greedy function c(e): incremental gain in objective_function when adding one
      POI to a pair. Because CRITIC/IQC are recomputed globally on every call, the
      gain is solution-dependent (this is the GRASP "adaptive" aspect).

The objective is non-separable and each evaluation recomputes CRITIC weights, so
construction uses *sampled greedy* (a random subset of pairs per step) and local
search runs under an evaluation budget. Both are tunable via the constants below.

Parallelism and caching
------------------------
The multi-start loop is embarrassingly parallel: each seed runs an independent
construction + local search. We dispatch one seed per worker process across
``max(cpu_count() - 1, 1)`` cores (Resende & Ribeiro report near-linear speedups
for the independent-strategy parallelization of GRASP). Results are deterministic
regardless of the number of workers, since each seed owns its own RNG. Each worker
also keeps a per-seed in-memory cache that memoizes ``_evaluate`` by candidate
content, avoiding recomputation when the same candidate matrix recurs.
"""

import os
import random
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core import (
    build_final_indicator_matrix_nd,
    objective_function,
)
from ..core.types import MetaheuristicContext, ObjectiveStateND


# RCL threshold: 0 -> pure greedy, 1 -> pure random. 0.2 is the value
# recommended by Resende & Ribeiro as a good mean/variance trade-off.
RCL_ALPHA = 0.2
# Sampled-greedy: number of (hex, dimension) pairs evaluated per construction step.
CONSTRUCTION_SAMPLE_SIZE = 64
# Local search runs first-improving moves until no improvement or this many
# objective evaluations are spent (per seed). Bounds the runtime per restart.
LOCAL_SEARCH_MAX_EVALS = 4000
# Minimum objective gain to count as an improvement (IQC is rounded to 4 decimals).
IMPROVEMENT_EPS = 1e-9

# Run the multi-start loop in parallel (one seed per worker process).
GRASP_PARALLEL = True
# Worker count: None -> max(cpu_count() - 1, 1). Set an int to override.
GRASP_MAX_WORKERS: Optional[int] = None
# Memoize _evaluate by candidate content inside each worker (per seed).
GRASP_EVAL_CACHE = True
# Safety bound: clear the per-seed cache if it grows beyond this many entries.
GRASP_EVAL_CACHE_MAXSIZE = 500_000


def _resolve_workers(n_seeds: int) -> int:
    """Number of worker processes: cpu-1 by default, never more than the seeds."""
    if not GRASP_PARALLEL or n_seeds <= 1:
        return 1
    workers = GRASP_MAX_WORKERS if GRASP_MAX_WORKERS else max((os.cpu_count() or 1) - 1, 1)
    return max(1, min(int(workers), n_seeds))


def _evaluate(candidate_matrix: np.ndarray,
              objective_state: ObjectiveStateND,
              cache: Optional[dict] = None) -> float:
    """Score a candidate matrix: propagate impacts then return sum(IQC).

    When ``cache`` is provided, the value is memoized by the candidate's bytes,
    so identical candidate matrices are evaluated only once.
    """
    if cache is not None:
        key = candidate_matrix.tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached

    final_indicator_matrix = build_final_indicator_matrix_nd(
        candidate_matrix=candidate_matrix,
        objective_state=objective_state,
    )
    value = objective_function(final_indicator_matrix=final_indicator_matrix)['objective_value']

    if cache is not None:
        if len(cache) >= GRASP_EVAL_CACHE_MAXSIZE:
            cache.clear()
        cache[key] = value
    return value


def _greedy_randomized_construction(objective_state: ObjectiveStateND,
                                    elements: List[Tuple[int, int]],
                                    budget: int,
                                    alpha: float,
                                    sample_size: int,
                                    rng: random.Random,
                                    cache: Optional[dict] = None) -> Tuple[np.ndarray, float, List[float]]:
    """Build a feasible allocation, one POI at a time, via a sampled RCL.

    Returns the candidate matrix, its objective value, and the construction
    trace (objective after each of the `budget` insertions) used for
    convergence analysis.
    """
    n_hex = len(objective_state.h3_ids)
    n_dims = len(objective_state.candidate_dimensions)
    candidate = np.zeros((n_hex, n_dims), dtype=np.float64)
    current_value = _evaluate(candidate, objective_state, cache)
    trace: List[float] = []

    for _ in range(budget):
        if sample_size < len(elements):
            sampled = rng.sample(elements, sample_size)
        else:
            sampled = list(elements)

        gains = np.empty(len(sampled), dtype=np.float64)
        for i, (row, col) in enumerate(sampled):
            candidate[row, col] += 1.0
            gains[i] = _evaluate(candidate, objective_state, cache) - current_value
            candidate[row, col] -= 1.0

        g_max = float(gains.max())
        g_min = float(gains.min())
        # Maximization RCL: keep elements whose gain is within alpha of the best.
        threshold = g_max - alpha * (g_max - g_min)
        rcl = [i for i in range(len(sampled)) if gains[i] >= threshold]

        chosen = rng.choice(rcl)
        row, col = sampled[chosen]
        candidate[row, col] += 1.0
        # gains[chosen] = value(candidate_with_element) - current_value, so this is exact.
        current_value += float(gains[chosen])
        trace.append(current_value)

    return candidate, current_value, trace


def _local_search(candidate: np.ndarray,
                  objective_state: ObjectiveStateND,
                  elements: List[Tuple[int, int]],
                  rng: random.Random,
                  max_evals: int,
                  cache: Optional[dict] = None) -> Tuple[np.ndarray, float]:
    """First-improving search: move one POI between pairs while budget stays fixed."""
    current_value = _evaluate(candidate, objective_state, cache)
    evals = 0
    improved = True

    while improved and evals < max_evals:
        improved = False
        occupied = [(int(r), int(c)) for r, c in zip(*np.nonzero(candidate))]
        rng.shuffle(occupied)

        for (row, col) in occupied:
            destinations = list(elements)
            rng.shuffle(destinations)

            for (row2, col2) in destinations:
                if row2 == row and col2 == col:
                    continue
                if evals >= max_evals:
                    break

                candidate[row, col] -= 1.0
                candidate[row2, col2] += 1.0
                value = _evaluate(candidate, objective_state, cache)
                evals += 1

                if value > current_value + IMPROVEMENT_EPS:
                    current_value = value
                    improved = True
                    break  # first-improving: accept and restart the sweep
                # revert
                candidate[row, col] += 1.0
                candidate[row2, col2] -= 1.0

            if improved or evals >= max_evals:
                break

    return candidate, current_value


def _candidate_matrix_to_items(candidate_matrix: np.ndarray,
                               objective_state: ObjectiveStateND) -> List[Dict[str, object]]:
    """Convert a candidate matrix back to compact allocation items."""
    items: List[Dict[str, object]] = []
    rows, cols = np.nonzero(candidate_matrix)
    for row, col in zip(rows, cols):
        items.append({
            'h3_id': objective_state.h3_ids[int(row)],
            'dimension': objective_state.candidate_dimensions[int(col)],
            'quantity': int(candidate_matrix[row, col]),
        })
    return items


def _run_seed(seed: int,
              objective_state: ObjectiveStateND,
              elements: List[Tuple[int, int]],
              budget: int,
              alpha: float,
              sample_size: int,
              max_evals: int,
              use_cache: bool) -> Dict[str, object]:
    """One independent GRASP restart: construction + local search for a seed."""
    rng = random.Random(int(seed))
    cache: Optional[dict] = {} if use_cache else None
    start = perf_counter()
    candidate, construction_value, trace = _greedy_randomized_construction(
        objective_state=objective_state, elements=elements, budget=budget,
        alpha=alpha, sample_size=sample_size, rng=rng, cache=cache,
    )
    candidate, value = _local_search(
        candidate=candidate, objective_state=objective_state, elements=elements,
        rng=rng, max_evals=max_evals, cache=cache,
    )
    return {
        'seed': int(seed),
        'candidate': candidate,
        'construction_objective': construction_value,
        'local_search_objective': value,
        'runtime_seconds': perf_counter() - start,
        'distinct_pairs': int(np.count_nonzero(candidate)),
        'trace': trace,
    }


# Worker-process state, populated once per worker via the pool initializer so the
# (potentially large) objective_state is pickled once per worker, not per task.
_WORKER: Dict[str, object] = {}


def _init_worker(objective_state, elements, budget, alpha, sample_size, max_evals, use_cache) -> None:
    _WORKER.update({
        'objective_state': objective_state, 'elements': elements, 'budget': budget,
        'alpha': alpha, 'sample_size': sample_size, 'max_evals': max_evals,
        'use_cache': use_cache,
    })


def _seed_task(seed: int) -> Dict[str, object]:
    w = _WORKER
    return _run_seed(seed, w['objective_state'], w['elements'], w['budget'],
                     w['alpha'], w['sample_size'], w['max_evals'], w['use_cache'])


# BLAS/OpenMP thread knobs. Each worker already provides process-level parallelism,
# so we pin each worker's numpy to a single thread to avoid CPU oversubscription.
_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
)


def _run_all_seeds(seeds: List[int],
                   objective_state: ObjectiveStateND,
                   elements: List[Tuple[int, int]],
                   budget: int) -> List[Dict[str, object]]:
    """Run every seed, in parallel across cpu-1 workers when worthwhile."""
    workers = _resolve_workers(len(seeds))
    seed_list = [int(s) for s in seeds]

    if workers > 1:
        # Spawned workers inherit os.environ; set single-thread BLAS before the
        # pool starts so each worker imports numpy without extra threads.
        saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
        for k in _THREAD_ENV_VARS:
            os.environ[k] = '1'
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(objective_state, elements, budget, RCL_ALPHA,
                          CONSTRUCTION_SAMPLE_SIZE, LOCAL_SEARCH_MAX_EVALS, GRASP_EVAL_CACHE),
            ) as executor:
                results = list(executor.map(_seed_task, seed_list))
            print(f"[grasp] multi-start over {len(seed_list)} seeds on {workers} worker(s).")
            return results
        except Exception as exc:  # restricted env, spawn issues, etc. -> serial fallback
            print(f"[grasp] parallel execution unavailable ({exc}); running serially.")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return [
        _run_seed(seed, objective_state, elements, budget, RCL_ALPHA,
                  CONSTRUCTION_SAMPLE_SIZE, LOCAL_SEARCH_MAX_EVALS, GRASP_EVAL_CACHE)
        for seed in seed_list
    ]


def run_grasp(context: MetaheuristicContext) -> dict:
    """Multi-start GRASP over the shared spatial-time objective (maximize sum IQC)."""
    objective_state = context.objective_state_nd
    if objective_state is None:
        return {
            'method_code': context.method_code,
            'method_name': context.method_name,
            'status': 'error',
            'message': 'objective_state_nd is missing; cannot run GRASP.',
        }

    n_dims = len(objective_state.candidate_dimensions)
    source_row_indices = np.unique(objective_state.source_indices).tolist()
    if n_dims == 0 or not source_row_indices:
        return {
            'method_code': context.method_code,
            'method_name': context.method_name,
            'status': 'error',
            'message': 'No allocatable (source hexagon, dimension) pairs available.',
        }
    if context.budget <= 0:
        return {
            'method_code': context.method_code,
            'method_name': context.method_name,
            'status': 'error',
            'message': 'Budget must be greater than zero.',
        }

    elements = [(int(row), col) for row in source_row_indices for col in range(n_dims)]
    seeds = context.seeds if context.seeds else [0]

    n_hex = len(objective_state.h3_ids)
    baseline_eval = objective_function(
        final_indicator_matrix=build_final_indicator_matrix_nd(
            candidate_matrix=np.zeros((n_hex, n_dims)),
            objective_state=objective_state,
        )
    )
    baseline_value = baseline_eval['objective_value']

    total_start = perf_counter()
    results = _run_all_seeds(seeds, objective_state, elements, context.budget)
    total_runtime = perf_counter() - total_start

    # Sort by seed so the output is identical whether run serially or in parallel.
    results.sort(key=lambda r: r['seed'])
    # Best solution: max objective; ties broken by smallest seed (results are sorted).
    best = max(results, key=lambda r: r['local_search_objective'])
    best_candidate = best['candidate']
    best_seed = best['seed']
    best_value = best['local_search_objective']
    workers_used = _resolve_workers(len(seeds))

    per_seed: List[Dict[str, object]] = [{
        'seed': r['seed'],
        'construction_objective': r['construction_objective'],
        'local_search_objective': r['local_search_objective'],
        'local_search_gain': r['local_search_objective'] - r['construction_objective'],
        'improvement_over_baseline': r['local_search_objective'] - baseline_value,
        'runtime_seconds': r['runtime_seconds'],
        'distinct_pairs': r['distinct_pairs'],
        'trace': r['trace'],
    } for r in results]

    best_items = _candidate_matrix_to_items(best_candidate, objective_state)
    best_final_matrix = build_final_indicator_matrix_nd(best_candidate, objective_state)
    best_eval = objective_function(final_indicator_matrix=best_final_matrix)

    statistics = {
        'baseline_objective': baseline_value,
        'best_objective': best_value,
        'best_seed': best_seed,
        'runtime_seconds_total': total_runtime,
        'per_seed': per_seed,
        'h3_ids': list(objective_state.h3_ids),
        'indicator_columns': list(objective_state.indicator_columns),
        'iqc_baseline': baseline_eval['iqc_values'],
        'iqc_optimized': best_eval['iqc_values'],
        'critic_weights_baseline': baseline_eval['critic_weights'],
        'critic_weights_optimized': best_eval['critic_weights'],
        'indicator_matrix_baseline': objective_state.baseline_matrix,
        'indicator_matrix_optimized': best_final_matrix,
        'parameters': {
            'rcl_alpha': RCL_ALPHA,
            'construction_sample_size': CONSTRUCTION_SAMPLE_SIZE,
            'local_search_max_evals': LOCAL_SEARCH_MAX_EVALS,
            'improvement_eps': IMPROVEMENT_EPS,
            'parallel_workers': workers_used,
            'eval_cache': GRASP_EVAL_CACHE,
        },
        'instance': {
            'n_hexagons': n_hex,
            'n_source_hexagons': len(source_row_indices),
            'n_candidate_dimensions': n_dims,
            'candidate_dimensions': list(objective_state.candidate_dimensions),
        },
    }

    return {
        'method_code': context.method_code,
        'method_name': context.method_name,
        'status': 'ok',
        'best_objective_value': best_value,
        'best_solution_summary': {
            'seed': best_seed,
            'rcl_alpha': RCL_ALPHA,
            'restarts': len(seeds),
            'allocated_pois': int(sum(item['quantity'] for item in best_items)),
            'distinct_pairs': len(best_items),
            'baseline_objective': baseline_value,
            'improvement': best_value - baseline_value,
            'allocation': best_items,
        },
        'statistics': statistics,
        'message': (
            f'GRASP finished: {len(seeds)} restarts, best sum(IQC)={best_value:.4f} '
            f'(baseline {baseline_value:.4f}, +{best_value - baseline_value:.4f}).'
        ),
    }
