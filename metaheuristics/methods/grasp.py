"""Classic GRASP (Greedy Randomized Adaptive Search Procedure).

Faithful to Resende & Ribeiro (2008), Figures 5 and 6:
each GRASP iteration performs a greedy randomized construction followed by a
local search, keeping the best solution over all iterations.

Design notes for this problem:
- Ground-set element = placing one POI unit at a (source hexagon, dimension) pair.
  The budget is therefore the number of construction steps (one POI per step),
  so every constructed solution has exactly ``budget`` POIs, matching the ILS
  solution space and its budget-conservation invariant.
- The objective ``sum(IQC)`` is global/non-separable (CRITIC re-normalizes over
  the whole matrix), so there is no cheap exact incremental cost. The
  construction uses a fast, fully vectorized *surrogate* greedy score
  (current CRITIC weight x alpha reach / current dimension range) to build the
  RCL, spending NO objective-evaluation budget inside the construction. Real
  ``objective_function`` evaluations are counted only for the completed solution
  and for the local search steps, exactly like ILS counts evaluations. Both
  methods are therefore bounded by the same ``max_evaluations`` budget.
- The local search operator is imported verbatim from the ILS implementation
  (first-improvement, dimension-preserving relocation), so the experimental
  difference between the two methods is isolated to construction + multistart
  (GRASP) vs. perturbation (ILS).
- RCL threshold parameter alpha is drawn uniformly at random in [0, 1] at each
  GRASP iteration (Resende & Ribeiro recommend against a single fixed alpha).

Outputs mirror the ILS artifact layout under ``results/grasp/...`` so both
methods can be compared directly.
"""

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.evaluation import _compute_critic_weights_numpy
from ..core.types import MetaheuristicContext, ObjectiveStateND
from .debug_nd_io import save_nd_debug_matrices
# Reuse the exact ILS algorithmic primitives so the local search operator and
# bookkeeping helpers are identical across both methods.
from .ils import (
    _build_dataset_tag,
    _candidate_matrix_to_allocation_items,
    _determine_stopping_reason,
    _evaluate_candidate_matrix,
    _resolve_allowed_source_rows,
    _run_local_search_first_improvement,
    _slugify,
)


@dataclass(frozen=True)
class GRASPRuntimeConfig:
    """Runtime configuration for GRASP experiments (mirrors ILS knobs)."""
    max_evaluations: int = 30000
    max_iterations: int = 500
    max_no_improve: int = 80
    local_search_neighbor_sample: int = 64
    local_search_max_steps: int = 25
    alpha_mode: str = "random"           # "random" -> U[0,1] per iteration; "fixed" -> alpha_fixed
    alpha_fixed: float = 0.2
    range_epsilon: float = 1e-9          # floor for dimension range in the surrogate score
    objective_epsilon: float = 1e-12
    experiment_mode: bool = True
    debug_mode: bool = False
    save_best_matrix_npz: bool = False
    log_enabled: bool = True
    log_every_iterations: int = 10
    log_only_improvements: bool = False
    seed_parallel_workers: int = 1


# --------------------------------------------------------------------------- #
# Construction surrogate                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ConstructionSurrogate:
    """Precompiled constants for the fast greedy-randomized construction."""
    base_weights: np.ndarray                       # CRITIC weight per candidate dimension
    reach_row: np.ndarray                          # sum of alpha impact leaving each hex row
    candidate_to_indicator_indices: np.ndarray     # candidate dim -> indicator column index
    source_targets: Dict[int, np.ndarray]          # source row -> unique target row indices
    source_alphas: Dict[int, np.ndarray]           # source row -> alpha per target


def _build_construction_surrogate(objective_state: ObjectiveStateND) -> _ConstructionSurrogate:
    baseline = objective_state.baseline_matrix
    n_hex = baseline.shape[0]

    # CRITIC weights of the empty solution (every GRASP construction starts empty),
    # restricted to candidate dimensions. This is a fast NumPy computation, not an
    # objective evaluation, so it is not counted against max_evaluations.
    full_weights = _compute_critic_weights_numpy(baseline)
    base_weights = full_weights[objective_state.candidate_to_indicator_indices].astype(np.float64, copy=True)

    reach_row = np.zeros(n_hex, dtype=np.float64)
    np.add.at(reach_row, objective_state.source_indices, objective_state.alpha_values)

    # Group impact rows by source hexagon (pairs are already unique per source-target).
    order = np.argsort(objective_state.source_indices, kind="stable")
    sorted_sources = objective_state.source_indices[order]
    sorted_targets = objective_state.target_indices[order]
    sorted_alphas = objective_state.alpha_values[order]

    source_targets: Dict[int, np.ndarray] = {}
    source_alphas: Dict[int, np.ndarray] = {}
    if sorted_sources.size > 0:
        unique_sources, start_positions = np.unique(sorted_sources, return_index=True)
        boundaries = list(start_positions) + [sorted_sources.size]
        for i, source_row in enumerate(unique_sources):
            lo = int(boundaries[i])
            hi = int(boundaries[i + 1])
            source_targets[int(source_row)] = sorted_targets[lo:hi].astype(np.int64, copy=True)
            source_alphas[int(source_row)] = sorted_alphas[lo:hi].astype(np.float64, copy=True)

    return _ConstructionSurrogate(
        base_weights=base_weights,
        reach_row=reach_row,
        candidate_to_indicator_indices=objective_state.candidate_to_indicator_indices.astype(np.int64, copy=True),
        source_targets=source_targets,
        source_alphas=source_alphas,
    )


def _greedy_randomized_construction(objective_state: ObjectiveStateND,
                                    surrogate: _ConstructionSurrogate,
                                    random_generator: np.random.Generator,
                                    allowed_source_rows: np.ndarray,
                                    budget: int,
                                    alpha: float,
                                    range_epsilon: float) -> Tuple[np.ndarray, float]:
    """Build one solution with the GRASP greedy randomized construction (Fig. 6).

    Returns the candidate matrix and the mean RCL size over the construction steps.
    """
    n_hex = objective_state.baseline_matrix.shape[0]
    n_dims = len(objective_state.candidate_dimensions)
    indicator_cols = surrogate.candidate_to_indicator_indices

    final_matrix = objective_state.baseline_matrix.copy()
    candidate_matrix = np.zeros((n_hex, n_dims), dtype=np.float64)

    allowed_rows = allowed_source_rows.astype(np.int64, copy=False)
    reach_allowed = surrogate.reach_row[allowed_rows]  # (n_allowed,)

    rcl_sizes: List[int] = []
    for _ in range(int(budget)):
        # Current dimension ranges drive diminishing returns as dimensions saturate.
        sub_matrix = final_matrix[:, indicator_cols]
        col_min = sub_matrix.min(axis=0)
        col_max = sub_matrix.max(axis=0)
        col_range = col_max - col_min
        range_safe = np.where(col_range > range_epsilon, col_range, range_epsilon)
        col_factor = surrogate.base_weights / range_safe  # (n_dims,)

        # Separable surrogate greedy score over (allowed source row, dimension).
        score_grid = reach_allowed[:, None] * col_factor[None, :]  # (n_allowed, n_dims)

        score_max = float(score_grid.max())
        score_min = float(score_grid.min())
        threshold = score_max - alpha * (score_max - score_min)
        rcl_mask = score_grid >= (threshold - 1e-15)
        rcl_positions = np.argwhere(rcl_mask)
        rcl_sizes.append(int(rcl_positions.shape[0]))

        chosen = int(random_generator.integers(0, rcl_positions.shape[0]))
        local_row = int(rcl_positions[chosen, 0])
        dim_idx = int(rcl_positions[chosen, 1])
        source_row = int(allowed_rows[local_row])

        candidate_matrix[source_row, dim_idx] += 1.0
        target_rows = surrogate.source_targets.get(source_row)
        if target_rows is not None and target_rows.size > 0:
            final_matrix[target_rows, int(indicator_cols[dim_idx])] += surrogate.source_alphas[source_row]

    mean_rcl_size = float(np.mean(rcl_sizes)) if rcl_sizes else 0.0
    return candidate_matrix, mean_rcl_size


# --------------------------------------------------------------------------- #
# Experiment directory / artifact helpers (mirror ILS, under results/grasp)    #
# --------------------------------------------------------------------------- #
def _build_experiment_directories(context: MetaheuristicContext) -> Dict[str, Path]:
    location_slug = _slugify(context.location)
    profile_slug = _slugify(context.walking_profile)
    dataset_tag = _slugify(_build_dataset_tag(context))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_budget{context.budget}_seeds{len(context.seeds)}"

    root_dir = Path("results") / "grasp" / location_slug / profile_slug / dataset_tag / run_id
    config_dir = root_dir / "config"
    summary_dir = root_dir / "summary"
    seed_runs_dir = root_dir / "seed_runs"

    config_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    seed_runs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root_dir": root_dir,
        "config_dir": config_dir,
        "summary_dir": summary_dir,
        "seed_runs_dir": seed_runs_dir,
    }


def _save_experiment_config(experiment_directories: Dict[str, Path],
                            context: MetaheuristicContext,
                            config: GRASPRuntimeConfig) -> Path:
    experiment_config = {
        "method_code": context.method_code,
        "method_name": context.method_name,
        "walking_profile": context.walking_profile,
        "budget": int(context.budget),
        "seeds": [int(seed) for seed in context.seeds],
        "dimensions": list(context.dimensions),
        "source_hexagon_count": len(context.source_hex_ids),
        "baseline_iqc_total": context.baseline_iqc_total,
        "runtime_config": asdict(config),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    config_path = experiment_directories["config_dir"] / "experiment_config.json"
    with config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(experiment_config, file_obj, ensure_ascii=False, indent=2)
    return config_path


def _save_per_seed_artifacts(experiment_directories: Dict[str, Path],
                             per_seed_result: Dict[str, object],
                             objective_state: ObjectiveStateND,
                             save_best_matrix_npz: bool) -> Dict[str, str]:
    seed_value = int(per_seed_result["seed"])
    seed_dir = experiment_directories["seed_runs_dir"] / f"seed_{seed_value:06d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = seed_dir / "trajectory.csv"
    pd.DataFrame(per_seed_result["trajectory"]).to_csv(trajectory_path, index=False, encoding="utf-8")

    best_allocation_path = seed_dir / "best_allocation.csv"
    pd.DataFrame(per_seed_result["best_allocation_items"]).to_csv(best_allocation_path, index=False, encoding="utf-8")

    run_metrics = {
        "seed": seed_value,
        "initial_objective_value": float(per_seed_result["initial_objective_value"]),
        "best_objective_value": float(per_seed_result["best_objective_value"]),
        "delta_abs_vs_baseline": per_seed_result["delta_abs_vs_baseline"],
        "delta_pct_vs_baseline": per_seed_result["delta_pct_vs_baseline"],
        "iterations": int(per_seed_result["iterations"]),
        "evaluations": int(per_seed_result["evaluations"]),
        "runtime_seconds": float(per_seed_result["runtime_seconds"]),
        "improvement_iterations": int(per_seed_result["improvement_iterations"]),
        "stopping_reason": per_seed_result["stopping_reason"],
        "best_alpha": per_seed_result["best_alpha"],
        "mean_rcl_size": per_seed_result["mean_rcl_size"],
        "construction_best_objective_value": per_seed_result["construction_best_objective_value"],
        "profiling": per_seed_result.get("profiling", {}),
    }
    run_metrics_path = seed_dir / "run_metrics.json"
    with run_metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(run_metrics, file_obj, ensure_ascii=False, indent=2)

    best_matrix_path = None
    if save_best_matrix_npz:
        best_matrix_path = seed_dir / "best_candidate_matrix.npz"
        np.savez_compressed(
            best_matrix_path,
            candidate_matrix=np.asarray(per_seed_result["best_candidate_matrix"], dtype=np.float64),
            h3_ids=np.asarray(objective_state.h3_ids),
            candidate_dimensions=np.asarray(objective_state.candidate_dimensions),
        )

    output_paths = {
        "seed_dir": str(seed_dir.resolve()),
        "trajectory_file": str(trajectory_path.resolve()),
        "best_allocation_file": str(best_allocation_path.resolve()),
        "run_metrics_file": str(run_metrics_path.resolve()),
    }
    if best_matrix_path is not None:
        output_paths["best_candidate_matrix_file"] = str(best_matrix_path.resolve())
    return output_paths


def _safe_iqr(values: List[float]) -> float:
    if not values:
        return 0.0
    q75, q25 = np.percentile(np.asarray(values, dtype=np.float64), [75, 25])
    return float(q75 - q25)


def _compute_summary_statistics(per_seed_run_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    if not per_seed_run_summaries:
        return {
            "n_runs": 0,
            "best_objective_mean": None,
            "best_objective_median": None,
            "best_objective_std": None,
            "best_objective_min": None,
            "best_objective_max": None,
            "best_objective_iqr": None,
            "runtime_seconds_mean": None,
            "evaluations_mean": None,
            "profiling_means": {},
        }

    objective_values = [float(item["best_objective_value"]) for item in per_seed_run_summaries]
    runtime_values = [float(item["runtime_seconds"]) for item in per_seed_run_summaries]
    evaluation_values = [int(item["evaluations"]) for item in per_seed_run_summaries]
    profile_mean_fields = [
        "profile_total_runtime_s",
        "profile_construction_total_s",
        "profile_candidate_eval_total_s",
        "profile_local_search_total_s",
        "profile_evaluation_total_s",
        "profile_evaluation_calls",
        "profile_evaluation_avg_s",
        "profile_local_search_calls",
    ]
    profiling_means: Dict[str, float] = {}
    for field_name in profile_mean_fields:
        field_values = [
            float(item[field_name])
            for item in per_seed_run_summaries
            if item.get(field_name) is not None
        ]
        if field_values:
            profiling_means[field_name] = float(np.mean(field_values))

    return {
        "n_runs": len(per_seed_run_summaries),
        "best_objective_mean": float(np.mean(objective_values)),
        "best_objective_median": float(np.median(objective_values)),
        "best_objective_std": float(np.std(objective_values, ddof=1)) if len(objective_values) > 1 else 0.0,
        "best_objective_min": float(np.min(objective_values)),
        "best_objective_max": float(np.max(objective_values)),
        "best_objective_iqr": _safe_iqr(objective_values),
        "runtime_seconds_mean": float(np.mean(runtime_values)),
        "evaluations_mean": float(np.mean(evaluation_values)),
        "profiling_means": profiling_means,
    }


def _save_experiment_summary_artifacts(experiment_directories: Dict[str, Path],
                                       per_seed_run_summaries: List[Dict[str, object]],
                                       summary_stats: Dict[str, object],
                                       global_best_seed_result: Dict[str, object]) -> Dict[str, str]:
    summary_dir = experiment_directories["summary_dir"]

    runs_summary_path = summary_dir / "runs_summary.csv"
    pd.DataFrame(per_seed_run_summaries).to_csv(runs_summary_path, index=False, encoding="utf-8")

    summary_stats_path = summary_dir / "summary_stats.json"
    with summary_stats_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary_stats, file_obj, ensure_ascii=False, indent=2)

    global_best_allocation_path = summary_dir / "global_best_allocation.csv"
    pd.DataFrame(global_best_seed_result["best_allocation_items"]).to_csv(
        global_best_allocation_path,
        index=False,
        encoding="utf-8",
    )

    return {
        "runs_summary_file": str(runs_summary_path.resolve()),
        "summary_stats_file": str(summary_stats_path.resolve()),
        "global_best_allocation_file": str(global_best_allocation_path.resolve()),
    }


def _save_method_result_artifact(experiment_directories: Dict[str, Path],
                                 method_result_payload: Dict[str, object]) -> str:
    method_result_path = experiment_directories["summary_dir"] / "method_result.json"
    with method_result_path.open("w", encoding="utf-8") as file_obj:
        json.dump(method_result_payload, file_obj, ensure_ascii=False, indent=2)
    return str(method_result_path.resolve())


def _resolve_seed_parallel_workers(seed_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, cpu_count - 3)
    raw_workers = os.getenv("GRASP_SEED_PARALLEL_WORKERS", os.getenv("GRASP_SEED_WORKERS"))
    requested_workers = default_workers
    if raw_workers is not None and str(raw_workers).strip() != "":
        requested_workers = int(raw_workers)
    return max(1, min(int(seed_count), int(requested_workers)))


def _build_per_seed_run_summary(seed_result: Dict[str, object],
                                seed_artifact_paths: Dict[str, str]) -> Dict[str, object]:
    seed_profile = seed_result.get("profiling", {})
    return {
        "seed": seed_result["seed"],
        "initial_objective_value": seed_result["initial_objective_value"],
        "best_objective_value": seed_result["best_objective_value"],
        "delta_abs_vs_baseline": seed_result["delta_abs_vs_baseline"],
        "delta_pct_vs_baseline": seed_result["delta_pct_vs_baseline"],
        "iterations": seed_result["iterations"],
        "evaluations": seed_result["evaluations"],
        "runtime_seconds": seed_result["runtime_seconds"],
        "improvement_iterations": seed_result["improvement_iterations"],
        "stopping_reason": seed_result["stopping_reason"],
        "best_alpha": seed_result["best_alpha"],
        "mean_rcl_size": seed_result["mean_rcl_size"],
        "construction_best_objective_value": seed_result["construction_best_objective_value"],
        "seed_dir": seed_artifact_paths.get("seed_dir"),
        "profile_total_runtime_s": seed_profile.get("total_runtime_s"),
        "profile_construction_total_s": seed_profile.get("construction_total_s"),
        "profile_candidate_eval_total_s": seed_profile.get("candidate_eval_total_s"),
        "profile_local_search_total_s": seed_profile.get("local_search_total_s"),
        "profile_evaluation_total_s": seed_profile.get("evaluation_total_s"),
        "profile_evaluation_calls": seed_profile.get("evaluation_calls"),
        "profile_evaluation_avg_s": seed_profile.get("evaluation_avg_s"),
        "profile_local_search_calls": seed_profile.get("local_search_calls"),
    }


# --------------------------------------------------------------------------- #
# Per-seed GRASP execution                                                     #
# --------------------------------------------------------------------------- #
def _run_single_seed_grasp(seed_value: int,
                           context: MetaheuristicContext,
                           objective_state: ObjectiveStateND,
                           surrogate: _ConstructionSurrogate,
                           config: GRASPRuntimeConfig) -> Dict[str, object]:
    random_generator = np.random.default_rng(seed_value)
    start_time = perf_counter()

    allowed_source_rows = _resolve_allowed_source_rows(context, objective_state)
    if allowed_source_rows.size == 0:
        raise ValueError("No valid source rows available for GRASP construction/local search moves.")

    budget = int(context.budget)
    alpha_mode = str(config.alpha_mode).strip().lower() or "random"
    baseline_value = context.baseline_iqc_total
    baseline_for_delta = float(baseline_value) if baseline_value is not None else None

    profiling_totals = {
        "construction_total_s": 0.0,
        "candidate_eval_total_s": 0.0,
        "local_search_total_s": 0.0,
        "evaluation_build_final_matrix_s": 0.0,
        "evaluation_objective_only_s": 0.0,
        "evaluation_total_s": 0.0,
        "evaluation_calls": 0,
        "local_search_calls": 0,
    }

    evaluations = 0
    iterations = 0
    improvement_iterations = 0
    no_improve_streak = 0

    best_objective_value = -math.inf
    best_candidate_matrix: Optional[np.ndarray] = None
    best_alpha: Optional[float] = None
    initial_objective_value: Optional[float] = None
    construction_best_objective_value = -math.inf
    rcl_size_accumulator: List[float] = []

    if config.log_enabled:
        print(
            f"[GRASP][seed={seed_value}] start "
            f"budget={budget} "
            f"baseline={baseline_for_delta if baseline_for_delta is not None else float('nan'):.4f} "
            f"alpha_mode={alpha_mode} alpha_fixed={config.alpha_fixed} "
            f"max_eval={config.max_evaluations} max_iter={config.max_iterations} "
            f"max_no_improve={config.max_no_improve}"
        )

    progress_log_every = max(1, int(config.log_every_iterations))
    trajectory_records: List[Dict[str, object]] = []

    while (
        evaluations < config.max_evaluations
        and iterations < config.max_iterations
        and no_improve_streak < config.max_no_improve
    ):
        iter_start = perf_counter()
        iterations += 1
        best_objective_before_iteration = best_objective_value

        if alpha_mode == "fixed":
            alpha = float(config.alpha_fixed)
        else:
            alpha = float(random_generator.random())
        alpha = min(1.0, max(0.0, alpha))

        # Phase 1: greedy randomized construction (no objective evaluations spent).
        construction_start = perf_counter()
        candidate_matrix, mean_rcl_size = _greedy_randomized_construction(
            objective_state=objective_state,
            surrogate=surrogate,
            random_generator=random_generator,
            allowed_source_rows=allowed_source_rows,
            budget=budget,
            alpha=alpha,
            range_epsilon=config.range_epsilon,
        )
        construction_elapsed = float(perf_counter() - construction_start)
        profiling_totals["construction_total_s"] += construction_elapsed
        rcl_size_accumulator.append(mean_rcl_size)

        # Evaluate the freshly constructed solution (counts as one evaluation).
        candidate_eval_start = perf_counter()
        construction_eval_result, construction_eval_timing = _evaluate_candidate_matrix(
            candidate_matrix, objective_state,
        )
        candidate_eval_elapsed = float(perf_counter() - candidate_eval_start)
        construction_objective_value = float(construction_eval_result["objective_value"])
        evaluations += 1
        profiling_totals["candidate_eval_total_s"] += candidate_eval_elapsed
        profiling_totals["evaluation_build_final_matrix_s"] += float(construction_eval_timing["build_final_matrix_s"])
        profiling_totals["evaluation_objective_only_s"] += float(construction_eval_timing["objective_eval_s"])
        profiling_totals["evaluation_total_s"] += float(construction_eval_timing["total_eval_s"])
        profiling_totals["evaluation_calls"] += 1
        if construction_objective_value > construction_best_objective_value:
            construction_best_objective_value = construction_objective_value

        # Phase 2: local search (identical operator to ILS), dimension-preserving.
        local_search_start = perf_counter()
        local_search_result = _run_local_search_first_improvement(
            candidate_matrix=candidate_matrix,
            objective_state=objective_state,
            random_generator=random_generator,
            current_objective_value=construction_objective_value,
            max_local_search_steps=config.local_search_max_steps,
            local_search_neighbor_sample=config.local_search_neighbor_sample,
            allowed_source_rows=allowed_source_rows,
            objective_epsilon=config.objective_epsilon,
            preserve_dimension=True,
        )
        local_search_elapsed = float(perf_counter() - local_search_start)
        after_ls_objective_value = float(local_search_result["objective_value"])
        evaluations += int(local_search_result["total_evaluations"])
        local_ls_profile = local_search_result["profiling"]
        profiling_totals["local_search_total_s"] += local_search_elapsed
        profiling_totals["evaluation_build_final_matrix_s"] += float(local_ls_profile["evaluation_build_final_matrix_s"])
        profiling_totals["evaluation_objective_only_s"] += float(local_ls_profile["evaluation_objective_s"])
        profiling_totals["evaluation_total_s"] += float(local_ls_profile["evaluation_total_s"])
        profiling_totals["evaluation_calls"] += int(local_search_result["total_evaluations"])
        profiling_totals["local_search_calls"] += 1

        if initial_objective_value is None:
            initial_objective_value = after_ls_objective_value

        improved = after_ls_objective_value > (best_objective_value + config.objective_epsilon)
        if improved:
            improvement_iterations += 1
            best_objective_value = after_ls_objective_value
            best_candidate_matrix = candidate_matrix.copy()
            best_alpha = alpha
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        elapsed_seconds = perf_counter() - start_time
        iter_total_elapsed = float(perf_counter() - iter_start)
        trajectory_records.append({
            "seed": int(seed_value),
            "iter": int(iterations),
            "eval_count": int(evaluations),
            "elapsed_s": float(elapsed_seconds),
            "alpha_used": float(alpha),
            "rcl_size_mean": float(mean_rcl_size),
            "construction_sum_iqc": float(construction_objective_value),
            "after_ls_sum_iqc": float(after_ls_objective_value),
            "best_sum_iqc": float(best_objective_value),
            "delta_vs_baseline": (best_objective_value - baseline_for_delta) if baseline_for_delta is not None else None,
            "delta_vs_best_prev": float(best_objective_value - best_objective_before_iteration)
            if math.isfinite(best_objective_before_iteration) else None,
            "improved": bool(improved),
            "no_improve_streak": int(no_improve_streak),
            "local_search_accepted_moves": int(local_search_result["accepted_local_moves"]),
            "t_construction_s": construction_elapsed,
            "t_candidate_eval_s": candidate_eval_elapsed,
            "t_local_search_s": local_search_elapsed,
            "t_iter_total_s": iter_total_elapsed,
        })

        should_log_progress = False
        if config.log_enabled:
            if config.log_only_improvements:
                should_log_progress = improved
            else:
                should_log_progress = (
                    improved
                    or iterations == 1
                    or (iterations % progress_log_every == 0)
                )
        if should_log_progress:
            print(
                f"[GRASP][seed={seed_value}] iter={iterations} eval={evaluations} "
                f"alpha={alpha:.3f} rcl={mean_rcl_size:.1f} "
                f"constr={construction_objective_value:.4f} after_ls={after_ls_objective_value:.4f} "
                f"best={best_objective_value:.4f} improved={improved} no_improve={no_improve_streak} "
                f"ls_acc={int(local_search_result['accepted_local_moves'])} "
                f"t_constr={construction_elapsed:.4f}s t_eval={candidate_eval_elapsed:.4f}s "
                f"t_ls={local_search_elapsed:.4f}s t_iter~{iter_total_elapsed:.4f}s"
            )

    runtime_seconds = float(perf_counter() - start_time)
    profiling_totals["total_runtime_s"] = runtime_seconds
    if profiling_totals["evaluation_calls"] > 0:
        profiling_totals["evaluation_avg_s"] = float(
            profiling_totals["evaluation_total_s"] / float(profiling_totals["evaluation_calls"])
        )
    else:
        profiling_totals["evaluation_avg_s"] = 0.0

    stopping_reason = _determine_stopping_reason(
        evaluations=evaluations,
        iterations=iterations,
        no_improve_streak=no_improve_streak,
        config=config,
    )

    if best_candidate_matrix is None:
        # Degenerate guard: no iteration ran (should not happen with a positive budget).
        best_candidate_matrix = np.zeros(
            (objective_state.baseline_matrix.shape[0], len(objective_state.candidate_dimensions)),
            dtype=np.float64,
        )
        best_objective_value = float(initial_objective_value) if initial_objective_value is not None else 0.0

    if config.log_enabled:
        print(
            f"[GRASP][seed={seed_value}] done "
            f"best={best_objective_value:.4f} eval={evaluations} iter={iterations} "
            f"improved={improvement_iterations} stop={stopping_reason} "
            f"best_alpha={best_alpha if best_alpha is not None else float('nan'):.3f} "
            f"runtime={runtime_seconds:.4f}s eval_total={profiling_totals['evaluation_total_s']:.4f}s "
            f"eval_calls={profiling_totals['evaluation_calls']}"
        )

    delta_abs_vs_baseline = None
    delta_pct_vs_baseline = None
    if baseline_for_delta is not None:
        delta_abs_vs_baseline = float(best_objective_value - baseline_for_delta)
        if not math.isclose(baseline_for_delta, 0.0):
            delta_pct_vs_baseline = float(100.0 * delta_abs_vs_baseline / baseline_for_delta)

    best_allocation_items = _candidate_matrix_to_allocation_items(
        candidate_matrix=best_candidate_matrix,
        objective_state=objective_state,
    )

    return {
        "seed": int(seed_value),
        "initial_objective_value": float(initial_objective_value) if initial_objective_value is not None else float(best_objective_value),
        "best_objective_value": float(best_objective_value),
        "delta_abs_vs_baseline": delta_abs_vs_baseline,
        "delta_pct_vs_baseline": delta_pct_vs_baseline,
        "iterations": int(iterations),
        "evaluations": int(evaluations),
        "runtime_seconds": runtime_seconds,
        "improvement_iterations": int(improvement_iterations),
        "stopping_reason": stopping_reason,
        "best_alpha": float(best_alpha) if best_alpha is not None else None,
        "mean_rcl_size": float(np.mean(rcl_size_accumulator)) if rcl_size_accumulator else 0.0,
        "construction_best_objective_value": float(construction_best_objective_value)
        if math.isfinite(construction_best_objective_value) else None,
        "trajectory": trajectory_records,
        "best_candidate_matrix": best_candidate_matrix,
        "best_allocation_items": best_allocation_items,
        "profiling": profiling_totals,
    }


# --------------------------------------------------------------------------- #
# Parallel seed workers                                                        #
# --------------------------------------------------------------------------- #
_WORKER_CONTEXT: Optional[MetaheuristicContext] = None
_WORKER_OBJECTIVE_STATE: Optional[ObjectiveStateND] = None
_WORKER_SURROGATE: Optional[_ConstructionSurrogate] = None
_WORKER_CONFIG: Optional[GRASPRuntimeConfig] = None


def _initialize_seed_worker(context: MetaheuristicContext,
                            objective_state: ObjectiveStateND,
                            surrogate: _ConstructionSurrogate,
                            config: GRASPRuntimeConfig) -> None:
    global _WORKER_CONTEXT
    global _WORKER_OBJECTIVE_STATE
    global _WORKER_SURROGATE
    global _WORKER_CONFIG
    _WORKER_CONTEXT = context
    _WORKER_OBJECTIVE_STATE = objective_state
    _WORKER_SURROGATE = surrogate
    _WORKER_CONFIG = config


def _run_single_seed_grasp_worker(seed_value: int) -> Dict[str, object]:
    if (
        _WORKER_CONTEXT is None
        or _WORKER_OBJECTIVE_STATE is None
        or _WORKER_SURROGATE is None
        or _WORKER_CONFIG is None
    ):
        raise RuntimeError("GRASP seed worker was not initialized.")
    return _run_single_seed_grasp(
        seed_value=int(seed_value),
        context=_WORKER_CONTEXT,
        objective_state=_WORKER_OBJECTIVE_STATE,
        surrogate=_WORKER_SURROGATE,
        config=_WORKER_CONFIG,
    )


# --------------------------------------------------------------------------- #
# Public runner                                                                #
# --------------------------------------------------------------------------- #
def run_grasp(context: MetaheuristicContext) -> dict:
    """Run a full classic GRASP experiment: one independent run per seed."""
    experiment_start_time = perf_counter()
    if context.objective_state_nd is None:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "objective_state_nd is required for GRASP ndarray execution.",
        }
    if not context.seeds:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "No seeds available for GRASP execution.",
        }

    seed_parallel_workers = _resolve_seed_parallel_workers(len(context.seeds))
    config = GRASPRuntimeConfig(
        max_evaluations=max(1, int(os.getenv("GRASP_MAX_EVALUATIONS", "30000"))),
        max_iterations=max(1, int(os.getenv("GRASP_MAX_ITERATIONS", "500"))),
        max_no_improve=max(1, int(os.getenv("GRASP_MAX_NO_IMPROVE", "80"))),
        local_search_neighbor_sample=max(1, int(os.getenv("GRASP_LS_NEIGHBOR_SAMPLE", "64"))),
        local_search_max_steps=max(1, int(os.getenv("GRASP_LS_MAX_STEPS", "25"))),
        alpha_mode=str(os.getenv("GRASP_ALPHA_MODE", "random")).strip().lower() or "random",
        alpha_fixed=float(os.getenv("GRASP_ALPHA_FIXED", "0.2")),
        experiment_mode=os.getenv("GRASP_EXPERIMENT_MODE", "1") != "0",
        debug_mode=os.getenv("GRASP_DEBUG_MODE", "0") == "1",
        save_best_matrix_npz=os.getenv("GRASP_SAVE_BEST_MATRIX_NPZ", "0") == "1",
        log_enabled=os.getenv("GRASP_LOG_ENABLED", "1") == "1",
        log_every_iterations=max(1, int(os.getenv("GRASP_LOG_EVERY", "10"))),
        log_only_improvements=os.getenv("GRASP_LOG_ONLY_IMPROVEMENTS", "0") == "1",
        seed_parallel_workers=seed_parallel_workers,
    )
    objective_state = context.objective_state_nd
    surrogate = _build_construction_surrogate(objective_state)

    experiment_directories = None
    config_file = None
    if config.experiment_mode:
        experiment_directories = _build_experiment_directories(context)
        config_file = _save_experiment_config(experiment_directories, context, config)

    seed_values = [int(seed) for seed in context.seeds]

    if config.debug_mode:
        # Persist the first construction per seed for audit, without spending eval budget.
        for seed_value in seed_values:
            debug_generator = np.random.default_rng(seed_value)
            allowed_rows = _resolve_allowed_source_rows(context, objective_state)
            debug_alpha = float(config.alpha_fixed) if config.alpha_mode == "fixed" else float(debug_generator.random())
            debug_candidate, _ = _greedy_randomized_construction(
                objective_state=objective_state,
                surrogate=surrogate,
                random_generator=debug_generator,
                allowed_source_rows=allowed_rows,
                budget=int(context.budget),
                alpha=min(1.0, max(0.0, debug_alpha)),
                range_epsilon=config.range_epsilon,
            )
            save_nd_debug_matrices(context=context, seed=seed_value, candidate_matrix=debug_candidate)

    seed_execution_start = perf_counter()
    if config.seed_parallel_workers > 1:
        if config.log_enabled:
            print(
                f"[GRASP] running {len(seed_values)} seeds "
                f"with {config.seed_parallel_workers} parallel workers"
            )
        with ProcessPoolExecutor(
            max_workers=config.seed_parallel_workers,
            initializer=_initialize_seed_worker,
            initargs=(context, objective_state, surrogate, config),
        ) as executor:
            per_seed_result_records = list(executor.map(_run_single_seed_grasp_worker, seed_values))
    else:
        per_seed_result_records = []
        for seed_value in seed_values:
            per_seed_result_records.append(
                _run_single_seed_grasp(
                    seed_value=seed_value,
                    context=context,
                    objective_state=objective_state,
                    surrogate=surrogate,
                    config=config,
                )
            )
    seed_execution_wall_seconds = float(perf_counter() - seed_execution_start)

    per_seed_run_summaries = []
    for seed_result in per_seed_result_records:
        seed_artifact_paths = {}
        if config.experiment_mode and experiment_directories is not None:
            seed_artifact_paths = _save_per_seed_artifacts(
                experiment_directories=experiment_directories,
                per_seed_result=seed_result,
                objective_state=objective_state,
                save_best_matrix_npz=config.save_best_matrix_npz,
            )
        seed_result["artifact_paths"] = seed_artifact_paths
        per_seed_run_summaries.append(
            _build_per_seed_run_summary(
                seed_result=seed_result,
                seed_artifact_paths=seed_artifact_paths,
            )
        )

    global_best_seed_result = max(
        per_seed_result_records,
        key=lambda item: float(item["best_objective_value"]),
    )
    seed_runtimes_total_seconds = float(
        sum(float(seed_result["runtime_seconds"]) for seed_result in per_seed_result_records)
    )
    experiment_runtime_seconds = float(perf_counter() - experiment_start_time)
    orchestration_runtime_seconds = float(experiment_runtime_seconds - seed_execution_wall_seconds)
    summary_stats = _compute_summary_statistics(per_seed_run_summaries)
    summary_stats["seed_runtimes_total_seconds"] = seed_runtimes_total_seconds
    summary_stats["seed_execution_wall_seconds"] = seed_execution_wall_seconds
    summary_stats["seed_parallel_workers"] = int(config.seed_parallel_workers)
    summary_stats["experiment_runtime_seconds"] = experiment_runtime_seconds
    summary_stats["orchestration_runtime_seconds"] = orchestration_runtime_seconds

    summary_artifacts = {}
    if config.experiment_mode and experiment_directories is not None:
        summary_artifacts = _save_experiment_summary_artifacts(
            experiment_directories=experiment_directories,
            per_seed_run_summaries=per_seed_run_summaries,
            summary_stats=summary_stats,
            global_best_seed_result=global_best_seed_result,
        )

    method_result_payload = {
        "method_code": context.method_code,
        "method_name": context.method_name,
        "status": "ok",
        "construction": "greedy_randomized_surrogate",
        "alpha_mode": config.alpha_mode,
        "n_runs": len(per_seed_result_records),
        "baseline_iqc_total": context.baseline_iqc_total,
        "experiment_runtime_seconds": experiment_runtime_seconds,
        "seed_runtimes_total_seconds": seed_runtimes_total_seconds,
        "seed_execution_wall_seconds": seed_execution_wall_seconds,
        "seed_parallel_workers": int(config.seed_parallel_workers),
        "orchestration_runtime_seconds": orchestration_runtime_seconds,
        "global_best_seed": int(global_best_seed_result["seed"]),
        "global_best_objective_value": float(global_best_seed_result["best_objective_value"]),
        "global_best_allocation_size": len(global_best_seed_result["best_allocation_items"]),
        "global_best_profiling": global_best_seed_result.get("profiling", {}),
        "summary_stats": summary_stats,
        "per_seed_results": per_seed_run_summaries,
        "output_dir": str(experiment_directories["root_dir"].resolve()) if experiment_directories is not None else None,
        "config_file": str(config_file.resolve()) if config_file is not None else None,
        **summary_artifacts,
        "message": "GRASP experiment completed with one full run per seed.",
    }
    if config.experiment_mode and experiment_directories is not None:
        method_result_payload["method_result_file"] = _save_method_result_artifact(
            experiment_directories=experiment_directories,
            method_result_payload=method_result_payload,
        )

    return method_result_payload
