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

from ..core import (
    allocation_items_to_candidate_matrix,
    build_final_indicator_matrix_nd,
    objective_function,
)
from ..core.types import MetaheuristicContext, ObjectiveStateND
from .debug_nd_io import save_nd_debug_matrices


@dataclass(frozen=True)
class ILSRuntimeConfig:
    """Runtime configuration for Iterated Local Search experiments."""
    max_evaluations: int = 30000
    max_iterations: int = 500
    max_no_improve: int = 80
    local_search_neighbor_sample: int = 64
    local_search_max_steps: int = 25
    perturbation_strength: int = 10
    perturbation_mode: str = "dimension_preserving_block"
    perturbation_block_size: int = 2
    adaptive_perturbation: bool = True
    perturbation_min_strength: int = 1
    perturbation_max_strength: Optional[int] = None
    perturbation_adapt_every: int = 10
    objective_epsilon: float = 1e-12
    acceptance_criterion: str = "better"
    experiment_mode: bool = True
    debug_mode: bool = False
    save_best_matrix_npz: bool = False
    log_enabled: bool = True
    log_every_iterations: int = 10
    log_only_improvements: bool = False
    log_on_acceptance: bool = False
    seed_parallel_workers: int = 1


def _slugify(value: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))
    safe = safe.strip('_').lower()
    return safe or "unknown"


def _build_dataset_tag(context: MetaheuristicContext) -> str:
    if context.objective_state_nd is not None:
        n_hexagons = len(context.objective_state_nd.h3_ids)
        n_impacts = int(context.objective_state_nd.source_indices.size)
    else:
        n_hexagons = len(context.df_walkability) if context.df_walkability is not None else 0
        n_impacts = len(context.df_hex_time_matrix) if context.df_hex_time_matrix is not None else 0
    return f"dataset_hex{n_hexagons}_impacts{n_impacts}"


def _build_experiment_directories(context: MetaheuristicContext) -> Dict[str, Path]:
    location_slug = _slugify(context.location)
    profile_slug = _slugify(context.walking_profile)
    dataset_tag = _slugify(_build_dataset_tag(context))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_budget{context.budget}_seeds{len(context.seeds)}"

    root_dir = Path("results") / "ils" / location_slug / profile_slug / dataset_tag / run_id
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


def _build_seed_initial_candidate_map(context: MetaheuristicContext) -> Dict[int, np.ndarray]:
    seed_to_candidate_matrix = {}
    if context.objective_state_nd is None:
        return seed_to_candidate_matrix

    for allocation_record in context.allocations:
        seed_value = int(allocation_record['seed'])
        allocation_items = allocation_record['allocation']
        candidate_matrix = allocation_items_to_candidate_matrix(
            allocation_items=allocation_items,
            objective_state=context.objective_state_nd,
        )
        seed_to_candidate_matrix[seed_value] = candidate_matrix
    return seed_to_candidate_matrix


def _candidate_matrix_to_allocation_items(candidate_matrix: np.ndarray,
                                          objective_state: ObjectiveStateND) -> List[Dict[str, object]]:
    non_zero_positions = np.argwhere(candidate_matrix > 0)
    allocation_items = []
    for row_idx, col_idx in non_zero_positions:
        quantity_value = candidate_matrix[row_idx, col_idx]
        quantity_int = int(round(float(quantity_value)))
        if quantity_int <= 0:
            continue
        allocation_items.append({
            "h3_id": objective_state.h3_ids[int(row_idx)],
            "dimension": objective_state.candidate_dimensions[int(col_idx)],
            "quantity": quantity_int,
        })
    allocation_items.sort(key=lambda item: (item["h3_id"], item["dimension"]))
    return allocation_items


def _sample_feasible_relocation_move(candidate_matrix: np.ndarray,
                                     random_generator: np.random.Generator,
                                     allowed_source_rows: np.ndarray,
                                     preserve_dimension: bool = False) -> Optional[Tuple[int, int, int, int]]:
    if allowed_source_rows.size == 0:
        return None

    source_positions = np.argwhere(candidate_matrix[allowed_source_rows, :] > 0)
    if source_positions.size == 0:
        return None

    n_cols = candidate_matrix.shape[1]
    n_allowed_rows = int(allowed_source_rows.shape[0])
    for _ in range(12):
        source_idx = int(random_generator.integers(0, len(source_positions)))
        source_row_local, source_col = source_positions[source_idx]
        source_row = int(allowed_source_rows[int(source_row_local)])
        target_row = int(allowed_source_rows[int(random_generator.integers(0, n_allowed_rows))])
        target_col = int(source_col) if preserve_dimension else int(random_generator.integers(0, n_cols))
        if source_row == target_row and source_col == target_col:
            continue
        return int(source_row), int(source_col), target_row, target_col
    return None


def _apply_relocation_move(candidate_matrix: np.ndarray,
                           source_row: int,
                           source_col: int,
                           target_row: int,
                           target_col: int) -> None:
    candidate_matrix[source_row, source_col] -= 1.0
    candidate_matrix[target_row, target_col] += 1.0


def _evaluate_candidate_matrix(candidate_matrix: np.ndarray,
                               objective_state: ObjectiveStateND) -> Tuple[Dict[str, object], Dict[str, float]]:
    build_start = perf_counter()
    final_indicator_matrix = build_final_indicator_matrix_nd(
        candidate_matrix=candidate_matrix,
        objective_state=objective_state,
    )
    build_end = perf_counter()
    objective_start = build_end
    eval_result = objective_function(
        final_indicator_matrix=final_indicator_matrix,
    )
    objective_end = perf_counter()
    return eval_result, {
        "build_final_matrix_s": float(build_end - build_start),
        "objective_eval_s": float(objective_end - objective_start),
        "total_eval_s": float(objective_end - build_start),
    }


def _apply_perturbation(candidate_matrix: np.ndarray,
                        random_generator: np.random.Generator,
                        perturbation_strength: int,
                        allowed_source_rows: np.ndarray,
                        preserve_dimension: bool = False) -> int:
    applied_moves = 0
    for _ in range(max(0, int(perturbation_strength))):
        relocation_move = _sample_feasible_relocation_move(
            candidate_matrix=candidate_matrix,
            random_generator=random_generator,
            allowed_source_rows=allowed_source_rows,
            preserve_dimension=preserve_dimension,
        )
        if relocation_move is None:
            break
        source_row, source_col, target_row, target_col = relocation_move
        _apply_relocation_move(candidate_matrix, source_row, source_col, target_row, target_col)
        applied_moves += 1
    return applied_moves


def _apply_dimension_preserving_block_perturbation(candidate_matrix: np.ndarray,
                                                    random_generator: np.random.Generator,
                                                    perturbation_strength: int,
                                                    allowed_source_rows: np.ndarray,
                                                    block_size: int = 2) -> int:
    n_blocks = max(0, int(perturbation_strength))
    block_moves = max(1, int(block_size))
    applied_moves = 0
    last_move: Optional[Tuple[int, int, int, int]] = None

    for _ in range(n_blocks):
        block_applied = 0
        block_attempts_limit = max(12, block_moves * 12)
        block_attempts = 0
        while block_applied < block_moves and block_attempts < block_attempts_limit:
            block_attempts += 1
            relocation_move = _sample_feasible_relocation_move(
                candidate_matrix=candidate_matrix,
                random_generator=random_generator,
                allowed_source_rows=allowed_source_rows,
                preserve_dimension=True,
            )
            if relocation_move is None:
                break

            source_row, source_col, target_row, target_col = relocation_move
            # Avoid trivial immediate inverse move inside the perturbation block.
            if (
                last_move is not None
                and source_row == last_move[2]
                and target_row == last_move[0]
                and source_col == last_move[1]
                and target_col == last_move[3]
            ):
                continue

            _apply_relocation_move(candidate_matrix, source_row, source_col, target_row, target_col)
            applied_moves += 1
            block_applied += 1
            last_move = relocation_move

        if block_applied == 0:
            break

    return applied_moves


def _run_local_search_first_improvement(candidate_matrix: np.ndarray,
                                        objective_state: ObjectiveStateND,
                                        random_generator: np.random.Generator,
                                        current_objective_value: float,
                                        max_local_search_steps: int,
                                        local_search_neighbor_sample: int,
                                        allowed_source_rows: np.ndarray,
                                        objective_epsilon: float,
                                        preserve_dimension: bool = False) -> Dict[str, object]:
    total_evaluations = 0
    accepted_local_moves = 0
    local_search_steps = 0
    objective_value = float(current_objective_value)
    local_search_start = perf_counter()
    eval_time_total = 0.0
    eval_build_time_total = 0.0
    eval_objective_time_total = 0.0
    move_apply_time_total = 0.0
    move_revert_time_total = 0.0
    neighbors_tested = 0

    for _ in range(max_local_search_steps):
        local_search_steps += 1
        found_improvement = False
        for _ in range(local_search_neighbor_sample):
            relocation_move = _sample_feasible_relocation_move(
                candidate_matrix=candidate_matrix,
                random_generator=random_generator,
                allowed_source_rows=allowed_source_rows,
                preserve_dimension=preserve_dimension,
            )
            if relocation_move is None:
                break

            source_row, source_col, target_row, target_col = relocation_move
            move_apply_start = perf_counter()
            _apply_relocation_move(candidate_matrix, source_row, source_col, target_row, target_col)
            move_apply_time_total += float(perf_counter() - move_apply_start)
            neighbors_tested += 1

            evaluation_result, eval_timing = _evaluate_candidate_matrix(candidate_matrix, objective_state)
            total_evaluations += 1
            eval_time_total += float(eval_timing["total_eval_s"])
            eval_build_time_total += float(eval_timing["build_final_matrix_s"])
            eval_objective_time_total += float(eval_timing["objective_eval_s"])
            trial_objective_value = float(evaluation_result['objective_value'])

            if trial_objective_value > (objective_value + objective_epsilon):
                objective_value = trial_objective_value
                accepted_local_moves += 1
                found_improvement = True
                break

            move_revert_start = perf_counter()
            _apply_relocation_move(candidate_matrix, target_row, target_col, source_row, source_col)
            move_revert_time_total += float(perf_counter() - move_revert_start)

        if not found_improvement:
            break

    local_search_total_time = float(perf_counter() - local_search_start)
    return {
        "objective_value": objective_value,
        "total_evaluations": total_evaluations,
        "accepted_local_moves": accepted_local_moves,
        "local_search_steps": local_search_steps,
        "profiling": {
            "local_search_total_s": local_search_total_time,
            "evaluation_total_s": eval_time_total,
            "evaluation_build_final_matrix_s": eval_build_time_total,
            "evaluation_objective_s": eval_objective_time_total,
            "move_apply_total_s": move_apply_time_total,
            "move_revert_total_s": move_revert_time_total,
            "neighbors_tested": int(neighbors_tested),
        },
    }


def _acceptance_criterion_better(candidate_objective_value: float,
                                 current_objective_value: float,
                                 objective_epsilon: float) -> bool:
    return candidate_objective_value > (current_objective_value + objective_epsilon)


def _resolve_allowed_source_rows(context: MetaheuristicContext,
                                 objective_state: ObjectiveStateND) -> np.ndarray:
    allowed_rows: List[int] = []
    for h3_id in context.source_hex_ids:
        row_idx = objective_state.h3_to_index.get(str(h3_id))
        if row_idx is not None:
            allowed_rows.append(int(row_idx))
    return np.asarray(sorted(set(allowed_rows)), dtype=np.int32)


def _validate_candidate_matrix_after_perturbation(candidate_matrix: np.ndarray,
                                                  total_sum_before: float,
                                                  column_sums_before: np.ndarray,
                                                  outside_allowed_rows: np.ndarray,
                                                  outside_allowed_matrix_before: Optional[np.ndarray],
                                                  perturbation_mode: str) -> None:
    total_sum_after = float(candidate_matrix.sum())
    if not np.isclose(total_sum_after, float(total_sum_before), rtol=0.0, atol=1e-9):
        raise ValueError(
            "ILS perturbation violated total budget conservation: "
            f"before={float(total_sum_before):.8f} after={total_sum_after:.8f}"
        )

    if np.any(candidate_matrix < -1e-12):
        min_value = float(np.min(candidate_matrix))
        raise ValueError(f"ILS perturbation produced negative allocation values (min={min_value:.8e}).")

    if perturbation_mode == "dimension_preserving_block":
        column_sums_after = candidate_matrix.sum(axis=0)
        if not np.allclose(column_sums_after, column_sums_before, rtol=0.0, atol=1e-9):
            raise ValueError("ILS perturbation violated per-dimension (column) budget conservation.")

    if outside_allowed_rows.size > 0 and outside_allowed_matrix_before is not None:
        outside_allowed_after = candidate_matrix[outside_allowed_rows, :]
        if not np.allclose(outside_allowed_after, outside_allowed_matrix_before, rtol=0.0, atol=1e-9):
            raise ValueError("ILS perturbation changed allocations outside allowed_source_rows.")


def _determine_stopping_reason(evaluations: int,
                               iterations: int,
                               no_improve_streak: int,
                               config: ILSRuntimeConfig) -> str:
    if evaluations >= config.max_evaluations:
        return "max_evaluations"
    if iterations >= config.max_iterations:
        return "max_iterations"
    if no_improve_streak >= config.max_no_improve:
        return "max_no_improve"
    return "loop_completed"


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
        "profile_initial_eval_s",
        "profile_initial_local_search_s",
        "profile_iteration_perturbation_s",
        "profile_iteration_candidate_eval_s",
        "profile_iteration_local_search_s",
        "profile_iteration_acceptance_update_s",
        "profile_iteration_bookkeeping_s",
        "profile_iteration_logging_s",
        "profile_evaluation_build_final_matrix_s",
        "profile_evaluation_objective_only_s",
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


def _save_experiment_config(experiment_directories: Dict[str, Path],
                            context: MetaheuristicContext,
                            config: ILSRuntimeConfig) -> Path:
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
    trajectory_df = pd.DataFrame(per_seed_result["trajectory"])
    trajectory_df.to_csv(trajectory_path, index=False, encoding="utf-8")

    best_allocation_path = seed_dir / "best_allocation.csv"
    allocation_df = pd.DataFrame(per_seed_result["best_allocation_items"])
    allocation_df.to_csv(best_allocation_path, index=False, encoding="utf-8")

    run_metrics = {
        "seed": seed_value,
        "initial_objective_value": float(per_seed_result["initial_objective_value"]),
        "best_objective_value": float(per_seed_result["best_objective_value"]),
        "delta_abs_vs_baseline": per_seed_result["delta_abs_vs_baseline"],
        "delta_pct_vs_baseline": per_seed_result["delta_pct_vs_baseline"],
        "iterations": int(per_seed_result["iterations"]),
        "evaluations": int(per_seed_result["evaluations"]),
        "runtime_seconds": float(per_seed_result["runtime_seconds"]),
        "accepted_iterations": int(per_seed_result["accepted_iterations"]),
        "improvement_iterations": int(per_seed_result["improvement_iterations"]),
        "stopping_reason": per_seed_result["stopping_reason"],
        "final_perturbation_strength": int(per_seed_result["final_perturbation_strength"]),
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
    raw_workers = os.getenv("ILS_SEED_PARALLEL_WORKERS", os.getenv("ILS_SEED_WORKERS"))
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
        "accepted_iterations": seed_result["accepted_iterations"],
        "improvement_iterations": seed_result["improvement_iterations"],
        "stopping_reason": seed_result["stopping_reason"],
        "final_perturbation_strength": seed_result["final_perturbation_strength"],
        "seed_dir": seed_artifact_paths.get("seed_dir"),
        "profile_total_runtime_s": seed_profile.get("total_runtime_s"),
        "profile_initial_eval_s": seed_profile.get("initial_eval_s"),
        "profile_initial_local_search_s": seed_profile.get("initial_local_search_s"),
        "profile_iteration_perturbation_s": seed_profile.get("iteration_perturbation_s"),
        "profile_iteration_candidate_eval_s": seed_profile.get("iteration_candidate_eval_s"),
        "profile_iteration_local_search_s": seed_profile.get("iteration_local_search_s"),
        "profile_iteration_acceptance_update_s": seed_profile.get("iteration_acceptance_update_s"),
        "profile_iteration_bookkeeping_s": seed_profile.get("iteration_bookkeeping_s"),
        "profile_iteration_logging_s": seed_profile.get("iteration_logging_s"),
        "profile_evaluation_build_final_matrix_s": seed_profile.get("evaluation_build_final_matrix_s"),
        "profile_evaluation_objective_only_s": seed_profile.get("evaluation_objective_only_s"),
        "profile_evaluation_total_s": seed_profile.get("evaluation_total_s"),
        "profile_evaluation_calls": seed_profile.get("evaluation_calls"),
        "profile_evaluation_avg_s": seed_profile.get("evaluation_avg_s"),
        "profile_local_search_calls": seed_profile.get("local_search_calls"),
    }


_WORKER_CONTEXT: Optional[MetaheuristicContext] = None
_WORKER_OBJECTIVE_STATE: Optional[ObjectiveStateND] = None
_WORKER_SEED_TO_INITIAL_CANDIDATE: Optional[Dict[int, np.ndarray]] = None
_WORKER_CONFIG: Optional[ILSRuntimeConfig] = None


def _initialize_seed_worker(context: MetaheuristicContext,
                            objective_state: ObjectiveStateND,
                            seed_to_initial_candidate: Dict[int, np.ndarray],
                            config: ILSRuntimeConfig) -> None:
    global _WORKER_CONTEXT
    global _WORKER_OBJECTIVE_STATE
    global _WORKER_SEED_TO_INITIAL_CANDIDATE
    global _WORKER_CONFIG
    _WORKER_CONTEXT = context
    _WORKER_OBJECTIVE_STATE = objective_state
    _WORKER_SEED_TO_INITIAL_CANDIDATE = seed_to_initial_candidate
    _WORKER_CONFIG = config


def _run_single_seed_ils_worker(seed_value: int) -> Dict[str, object]:
    if (
        _WORKER_CONTEXT is None
        or _WORKER_OBJECTIVE_STATE is None
        or _WORKER_SEED_TO_INITIAL_CANDIDATE is None
        or _WORKER_CONFIG is None
    ):
        raise RuntimeError("ILS seed worker was not initialized.")

    seed_int = int(seed_value)
    initial_candidate_matrix = _WORKER_SEED_TO_INITIAL_CANDIDATE[seed_int].copy()
    return _run_single_seed_ils(
        seed_value=seed_int,
        context=_WORKER_CONTEXT,
        objective_state=_WORKER_OBJECTIVE_STATE,
        initial_candidate_matrix=initial_candidate_matrix,
        config=_WORKER_CONFIG,
    )


def _run_single_seed_ils(seed_value: int,
                         context: MetaheuristicContext,
                         objective_state: ObjectiveStateND,
                         initial_candidate_matrix: np.ndarray,
                         config: ILSRuntimeConfig) -> Dict[str, object]:
    random_generator = np.random.default_rng(seed_value)
    start_time = perf_counter()
    allowed_source_rows = _resolve_allowed_source_rows(context, objective_state)
    if allowed_source_rows.size == 0:
        raise ValueError("No valid source rows available for ILS perturbation/local search moves.")

    # Pseudocode-aligned initialization:
    # s0 = GenerateInitialSolution
    # s* = LocalSearch(s0)
    current_candidate_matrix = np.asarray(initial_candidate_matrix, dtype=np.float64).copy()
    all_rows = np.arange(current_candidate_matrix.shape[0], dtype=np.int32)
    outside_allowed_rows = np.setdiff1d(all_rows, allowed_source_rows, assume_unique=True)
    initial_eval_start = perf_counter()
    initial_eval_result, initial_eval_timing = _evaluate_candidate_matrix(current_candidate_matrix, objective_state)
    initial_eval_elapsed = float(perf_counter() - initial_eval_start)
    initial_objective_value = float(initial_eval_result["objective_value"])
    evaluations = 1
    iterations = 0

    profiling_totals = {
        "initial_eval_s": initial_eval_elapsed,
        "initial_local_search_s": 0.0,
        "iteration_perturbation_s": 0.0,
        "iteration_candidate_eval_s": 0.0,
        "iteration_local_search_s": 0.0,
        "iteration_acceptance_update_s": 0.0,
        "iteration_bookkeeping_s": 0.0,
        "iteration_logging_s": 0.0,
        "evaluation_build_final_matrix_s": float(initial_eval_timing["build_final_matrix_s"]),
        "evaluation_objective_only_s": float(initial_eval_timing["objective_eval_s"]),
        "evaluation_total_s": float(initial_eval_timing["total_eval_s"]),
        "evaluation_calls": 1,
        "local_search_calls": 0,
    }

    initial_local_search_start = perf_counter()
    initial_local_search_result = _run_local_search_first_improvement(
        candidate_matrix=current_candidate_matrix,
        objective_state=objective_state,
        random_generator=random_generator,
        current_objective_value=initial_objective_value,
        max_local_search_steps=config.local_search_max_steps,
        local_search_neighbor_sample=config.local_search_neighbor_sample,
        allowed_source_rows=allowed_source_rows,
        objective_epsilon=config.objective_epsilon,
        preserve_dimension=True,
    )
    initial_local_search_elapsed = float(perf_counter() - initial_local_search_start)
    current_objective_value = float(initial_local_search_result["objective_value"])
    evaluations += int(initial_local_search_result["total_evaluations"])
    initial_ls_profile = initial_local_search_result["profiling"]
    profiling_totals["initial_local_search_s"] += initial_local_search_elapsed
    profiling_totals["evaluation_build_final_matrix_s"] += float(initial_ls_profile["evaluation_build_final_matrix_s"])
    profiling_totals["evaluation_objective_only_s"] += float(initial_ls_profile["evaluation_objective_s"])
    profiling_totals["evaluation_total_s"] += float(initial_ls_profile["evaluation_total_s"])
    profiling_totals["evaluation_calls"] += int(initial_local_search_result["total_evaluations"])
    profiling_totals["local_search_calls"] += 1

    best_candidate_matrix = current_candidate_matrix.copy()
    best_objective_value = current_objective_value

    accepted_iterations = 0
    improvement_iterations = 0
    no_improve_streak = 0

    perturbation_mode = str(config.perturbation_mode).strip().lower() or "dimension_preserving_block"
    perturb_block_size = max(1, int(config.perturbation_block_size))
    perturb_adapt_every = max(1, int(config.perturbation_adapt_every))
    perturb_min_strength = max(1, min(int(config.perturbation_min_strength), int(context.budget)))
    if config.perturbation_max_strength is None:
        perturb_max_strength = int(context.budget)
    else:
        perturb_max_strength = int(config.perturbation_max_strength)
    perturb_max_strength = max(perturb_min_strength, min(perturb_max_strength, int(context.budget)))
    perturb_base_strength = max(
        perturb_min_strength,
        min(int(config.perturbation_strength), perturb_max_strength),
    )
    perturb_current = perturb_base_strength
    progress_log_every = max(1, int(config.log_every_iterations))
    if config.log_enabled:
        print(
            f"[ILS][seed={seed_value}] start "
            f"initial={initial_objective_value:.4f} "
            f"after_ls={current_objective_value:.4f} "
            f"best={best_objective_value:.4f} "
            f"perturbation={perturb_current} "
            f"perturb_mode={perturbation_mode} "
            f"block_size={perturb_block_size} "
            f"adaptive={config.adaptive_perturbation} "
            f"acceptance={config.acceptance_criterion}"
        )

    baseline_value = context.baseline_iqc_total
    baseline_for_delta = float(baseline_value) if baseline_value is not None else None

    trajectory_records = [{
        "seed": int(seed_value),
        "iter": 0,
        "eval_count": evaluations,
        "elapsed_s": 0.0,
        "current_sum_iqc": current_objective_value,
        "best_sum_iqc": best_objective_value,
        "delta_vs_baseline": (best_objective_value - baseline_for_delta) if baseline_for_delta is not None else None,
        "delta_vs_best_prev": 0.0,
        "accepted": True,
        "improved": False,
        "perturbation_strength": int(perturb_current),
        "no_improve_streak": 0,
        "local_search_accepted_moves": int(initial_local_search_result["accepted_local_moves"]),
        "applied_perturbation_moves": 0,
        "t_initial_eval_s": initial_eval_elapsed,
        "t_initial_local_search_s": initial_local_search_elapsed,
        "t_perturb_s": 0.0,
        "t_candidate_eval_s": 0.0,
        "t_local_search_s": 0.0,
        "t_acceptance_s": 0.0,
        "t_bookkeeping_s": 0.0,
        "t_logging_s": 0.0,
        "t_iter_total_s": 0.0,
    }]

    while evaluations < config.max_evaluations and iterations < config.max_iterations and no_improve_streak < config.max_no_improve:
        iter_start = perf_counter()
        iterations += 1
        best_objective_before_iteration = best_objective_value

        candidate_matrix = current_candidate_matrix.copy()
        perturb_strength_used = int(perturb_current)
        perturb_total_before = 0.0
        perturb_column_sums_before: Optional[np.ndarray] = None
        outside_allowed_before: Optional[np.ndarray] = None
        if config.debug_mode:
            perturb_total_before = float(candidate_matrix.sum())
            perturb_column_sums_before = candidate_matrix.sum(axis=0).copy()
            outside_allowed_before = (
                candidate_matrix[outside_allowed_rows, :].copy()
                if outside_allowed_rows.size > 0
                else None
            )

        perturb_start = perf_counter()
        if perturbation_mode == "dimension_preserving_block":
            applied_perturbation_moves = _apply_dimension_preserving_block_perturbation(
                candidate_matrix=candidate_matrix,
                random_generator=random_generator,
                perturbation_strength=perturb_strength_used,
                allowed_source_rows=allowed_source_rows,
                block_size=perturb_block_size,
            )
        else:
            applied_perturbation_moves = _apply_perturbation(
                candidate_matrix=candidate_matrix,
                random_generator=random_generator,
                perturbation_strength=perturb_strength_used,
                allowed_source_rows=allowed_source_rows,
                preserve_dimension=True,
            )
        perturb_elapsed = float(perf_counter() - perturb_start)
        profiling_totals["iteration_perturbation_s"] += perturb_elapsed
        if config.debug_mode:
            _validate_candidate_matrix_after_perturbation(
                candidate_matrix=candidate_matrix,
                total_sum_before=perturb_total_before,
                column_sums_before=(
                    perturb_column_sums_before
                    if perturb_column_sums_before is not None
                    else np.zeros(candidate_matrix.shape[1], dtype=np.float64)
                ),
                outside_allowed_rows=outside_allowed_rows,
                outside_allowed_matrix_before=outside_allowed_before,
                perturbation_mode=perturbation_mode,
            )

        candidate_eval_start = perf_counter()
        candidate_eval_result, candidate_eval_timing = _evaluate_candidate_matrix(candidate_matrix, objective_state)
        candidate_eval_elapsed = float(perf_counter() - candidate_eval_start)
        candidate_objective_value = float(candidate_eval_result["objective_value"])
        evaluations += 1
        profiling_totals["iteration_candidate_eval_s"] += candidate_eval_elapsed
        profiling_totals["evaluation_build_final_matrix_s"] += float(candidate_eval_timing["build_final_matrix_s"])
        profiling_totals["evaluation_objective_only_s"] += float(candidate_eval_timing["objective_eval_s"])
        profiling_totals["evaluation_total_s"] += float(candidate_eval_timing["total_eval_s"])
        profiling_totals["evaluation_calls"] += 1

        local_search_start = perf_counter()
        local_search_result = _run_local_search_first_improvement(
            candidate_matrix=candidate_matrix,
            objective_state=objective_state,
            random_generator=random_generator,
            current_objective_value=candidate_objective_value,
            max_local_search_steps=config.local_search_max_steps,
            local_search_neighbor_sample=config.local_search_neighbor_sample,
            allowed_source_rows=allowed_source_rows,
            objective_epsilon=config.objective_epsilon,
            preserve_dimension=True,
        )
        local_search_elapsed = float(perf_counter() - local_search_start)
        candidate_objective_value = float(local_search_result["objective_value"])
        evaluations += int(local_search_result["total_evaluations"])
        local_ls_profile = local_search_result["profiling"]
        profiling_totals["iteration_local_search_s"] += local_search_elapsed
        profiling_totals["evaluation_build_final_matrix_s"] += float(local_ls_profile["evaluation_build_final_matrix_s"])
        profiling_totals["evaluation_objective_only_s"] += float(local_ls_profile["evaluation_objective_s"])
        profiling_totals["evaluation_total_s"] += float(local_ls_profile["evaluation_total_s"])
        profiling_totals["evaluation_calls"] += int(local_search_result["total_evaluations"])
        profiling_totals["local_search_calls"] += 1

        acceptance_start = perf_counter()
        accepted = False
        improved = False
        accepted = _acceptance_criterion_better(
            candidate_objective_value=candidate_objective_value,
            current_objective_value=current_objective_value,
            objective_epsilon=config.objective_epsilon,
        )

        if accepted:
            accepted_iterations += 1
            current_candidate_matrix = candidate_matrix
            current_objective_value = candidate_objective_value

        if current_objective_value > (best_objective_value + config.objective_epsilon):
            improvement_iterations += 1
            improved = True
            best_objective_value = current_objective_value
            best_candidate_matrix = current_candidate_matrix.copy()
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        if improved:
            perturb_current = perturb_base_strength
        elif config.adaptive_perturbation:
            if no_improve_streak > 0 and (no_improve_streak % perturb_adapt_every == 0):
                perturb_current = min(perturb_current + 1, perturb_max_strength)

        acceptance_elapsed = float(perf_counter() - acceptance_start)
        profiling_totals["iteration_acceptance_update_s"] += acceptance_elapsed

        bookkeeping_start = perf_counter()
        elapsed_seconds = perf_counter() - start_time
        trajectory_records.append({
            "seed": int(seed_value),
            "iter": int(iterations),
            "eval_count": int(evaluations),
            "elapsed_s": float(elapsed_seconds),
            "current_sum_iqc": float(current_objective_value),
            "best_sum_iqc": float(best_objective_value),
            "delta_vs_baseline": (best_objective_value - baseline_for_delta) if baseline_for_delta is not None else None,
            "delta_vs_best_prev": float(best_objective_value - best_objective_before_iteration),
            "accepted": bool(accepted),
            "improved": bool(improved),
            "perturbation_strength": int(perturb_strength_used),
            "no_improve_streak": int(no_improve_streak),
            "local_search_accepted_moves": int(local_search_result["accepted_local_moves"]),
            "applied_perturbation_moves": int(applied_perturbation_moves),
            "t_initial_eval_s": 0.0,
            "t_initial_local_search_s": 0.0,
            "t_perturb_s": perturb_elapsed,
            "t_candidate_eval_s": candidate_eval_elapsed,
            "t_local_search_s": local_search_elapsed,
            "t_acceptance_s": acceptance_elapsed,
            "t_bookkeeping_s": 0.0,
            "t_logging_s": 0.0,
            "t_iter_total_s": 0.0,
        })
        bookkeeping_elapsed = float(perf_counter() - bookkeeping_start)
        profiling_totals["iteration_bookkeeping_s"] += bookkeeping_elapsed
        trajectory_records[-1]["t_bookkeeping_s"] = bookkeeping_elapsed

        should_log_progress = False
        if config.log_enabled:
            if config.log_only_improvements:
                should_log_progress = improved
            else:
                should_log_progress = (
                    improved
                    or iterations == 1
                    or (iterations % progress_log_every == 0)
                    or (config.log_on_acceptance and accepted)
                )

        logging_elapsed = 0.0
        if should_log_progress:
            log_start = perf_counter()
            iter_elapsed_partial = float(perf_counter() - iter_start)
            print(
                f"[ILS][seed={seed_value}] iter={iterations} eval={evaluations} "
                f"candidate={candidate_objective_value:.4f} current={current_objective_value:.4f} "
                f"best={best_objective_value:.4f} perturbation={perturb_strength_used} "
                f"pert_mode={perturbation_mode} block_size={perturb_block_size} "
                f"pert_moves={applied_perturbation_moves} accepted={accepted} "
                f"improved={improved} no_improve={no_improve_streak} "
                f"ls_acc={int(local_search_result['accepted_local_moves'])} "
                f"t_pert={perturb_elapsed:.4f}s t_eval={candidate_eval_elapsed:.4f}s "
                f"t_ls={local_search_elapsed:.4f}s t_iter~{iter_elapsed_partial:.4f}s"
            )
            logging_elapsed = float(perf_counter() - log_start)
        profiling_totals["iteration_logging_s"] += logging_elapsed
        trajectory_records[-1]["t_logging_s"] = logging_elapsed
        trajectory_records[-1]["t_iter_total_s"] = float(perf_counter() - iter_start)

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
    if config.log_enabled:
        print(
            f"[ILS][seed={seed_value}] done "
            f"best={best_objective_value:.4f} eval={evaluations} iter={iterations} "
            f"accepted={accepted_iterations} improved={improvement_iterations} "
            f"stop={stopping_reason} "
            f"runtime={runtime_seconds:.4f}s eval_total={profiling_totals['evaluation_total_s']:.4f}s "
            f"eval_calls={profiling_totals['evaluation_calls']} "
            f"pert_mode={perturbation_mode} block_size={perturb_block_size}"
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
        "initial_objective_value": float(initial_objective_value),
        "best_objective_value": float(best_objective_value),
        "delta_abs_vs_baseline": delta_abs_vs_baseline,
        "delta_pct_vs_baseline": delta_pct_vs_baseline,
        "iterations": int(iterations),
        "evaluations": int(evaluations),
        "runtime_seconds": runtime_seconds,
        "accepted_iterations": int(accepted_iterations),
        "improvement_iterations": int(improvement_iterations),
        "stopping_reason": stopping_reason,
        "final_perturbation_strength": int(perturb_current),
        "trajectory": trajectory_records,
        "best_candidate_matrix": best_candidate_matrix,
        "best_allocation_items": best_allocation_items,
        "profiling": profiling_totals,
    }


def run_ils(context: MetaheuristicContext) -> dict:
    """Run full Iterated Local Search experiment across all seeds."""
    experiment_start_time = perf_counter()
    if context.objective_state_nd is None:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "objective_state_nd is required for ILS ndarray execution.",
        }
    if not context.allocations:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "No allocation candidates available.",
        }

    perturbation_mode = os.getenv("ILS_PERTURBATION_MODE", "dimension_preserving_block")
    perturbation_max_strength_raw = os.getenv("ILS_PERTURBATION_MAX_STRENGTH")
    perturbation_max_strength = None
    if perturbation_max_strength_raw is not None and str(perturbation_max_strength_raw).strip() != "":
        perturbation_max_strength = int(perturbation_max_strength_raw)
    seed_parallel_workers = _resolve_seed_parallel_workers(len(context.seeds))

    config = ILSRuntimeConfig(
        perturbation_strength=max(1, int(os.getenv("ILS_PERTURBATION_STRENGTH", "10"))),
        perturbation_mode=str(perturbation_mode).strip().lower() or "dimension_preserving_block",
        perturbation_block_size=max(1, int(os.getenv("ILS_PERTURBATION_BLOCK_SIZE", "2"))),
        adaptive_perturbation=os.getenv("ILS_ADAPTIVE_PERTURBATION", "1") != "0",
        perturbation_min_strength=max(1, int(os.getenv("ILS_PERTURBATION_MIN_STRENGTH", "1"))),
        perturbation_max_strength=perturbation_max_strength,
        perturbation_adapt_every=max(1, int(os.getenv("ILS_PERTURBATION_ADAPT_EVERY", "10"))),
        acceptance_criterion="better",
        experiment_mode=os.getenv("ILS_EXPERIMENT_MODE", "1") != "0",
        debug_mode=os.getenv("ILS_DEBUG_MODE", "0") == "1",
        save_best_matrix_npz=os.getenv("ILS_SAVE_BEST_MATRIX_NPZ", "0") == "1",
        log_enabled=os.getenv("ILS_LOG_ENABLED", "1") == "1",
        log_every_iterations=max(1, int(os.getenv("ILS_LOG_EVERY", "10"))),
        log_only_improvements=os.getenv("ILS_LOG_ONLY_IMPROVEMENTS", "0") == "1",
        log_on_acceptance=os.getenv("ILS_LOG_ON_ACCEPTANCE", "0") == "1",
        seed_parallel_workers=seed_parallel_workers,
    )
    objective_state = context.objective_state_nd

    seed_to_initial_candidate = _build_seed_initial_candidate_map(context)
    missing_initial_seeds = [seed for seed in context.seeds if int(seed) not in seed_to_initial_candidate]
    if missing_initial_seeds:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": f"Missing initial allocations for seeds: {missing_initial_seeds[:5]}",
        }

    experiment_directories = None
    config_file = None
    if config.experiment_mode:
        experiment_directories = _build_experiment_directories(context)
        config_file = _save_experiment_config(experiment_directories, context, config)

    per_seed_result_records = []
    per_seed_run_summaries = []
    seed_values = [int(seed) for seed in context.seeds]
    debug_files_by_seed: Dict[int, Dict[str, str]] = {}
    if config.debug_mode:
        for seed_value in seed_values:
            debug_files_by_seed[seed_value] = save_nd_debug_matrices(
                context=context,
                seed=seed_value,
                candidate_matrix=seed_to_initial_candidate[seed_value].copy(),
            )

    seed_execution_start = perf_counter()
    if config.seed_parallel_workers > 1:
        if config.log_enabled:
            print(
                f"[ILS] running {len(seed_values)} seeds "
                f"with {config.seed_parallel_workers} parallel workers"
            )
        with ProcessPoolExecutor(
            max_workers=config.seed_parallel_workers,
            initializer=_initialize_seed_worker,
            initargs=(context, objective_state, seed_to_initial_candidate, config),
        ) as executor:
            per_seed_result_records = list(executor.map(_run_single_seed_ils_worker, seed_values))
    else:
        for seed_value in seed_values:
            initial_candidate_matrix = seed_to_initial_candidate[seed_value].copy()
            seed_result = _run_single_seed_ils(
                seed_value=seed_value,
                context=context,
                objective_state=objective_state,
                initial_candidate_matrix=initial_candidate_matrix,
                config=config,
            )
            per_seed_result_records.append(seed_result)
    seed_execution_wall_seconds = float(perf_counter() - seed_execution_start)

    for seed_result in per_seed_result_records:
        seed_value = int(seed_result["seed"])
        seed_artifact_paths = {}
        if config.experiment_mode and experiment_directories is not None:
            seed_artifact_paths = _save_per_seed_artifacts(
                experiment_directories=experiment_directories,
                per_seed_result=seed_result,
                objective_state=objective_state,
                save_best_matrix_npz=config.save_best_matrix_npz,
            )
        seed_result["artifact_paths"] = seed_artifact_paths
        debug_files = debug_files_by_seed.get(seed_value, {})
        if debug_files:
            seed_result["debug_files"] = debug_files
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
        "acceptance_criterion": config.acceptance_criterion,
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
        "message": "ILS experiment completed with one full run per seed.",
    }
    if config.experiment_mode and experiment_directories is not None:
        method_result_payload["method_result_file"] = _save_method_result_artifact(
            experiment_directories=experiment_directories,
            method_result_payload=method_result_payload,
        )

    return method_result_payload
