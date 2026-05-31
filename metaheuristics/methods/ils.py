import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core import (
    allocation_items_to_candidate_matrix,
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
    perturbation_strength: int = 2
    objective_epsilon: float = 1e-12
    acceptance_criterion: str = "better"
    experiment_mode: bool = True
    debug_mode: bool = False
    save_best_matrix_npz: bool = False
    log_enabled: bool = True
    log_every_iterations: int = 10
    log_only_improvements: bool = False
    log_on_acceptance: bool = False


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
    profile_slug = _slugify(context.walking_profile)
    dataset_tag = _slugify(_build_dataset_tag(context))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_budget{context.budget}_seeds{len(context.seeds)}"

    root_dir = Path("results") / "ils" / profile_slug / dataset_tag / run_id
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
                                     allowed_source_rows: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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
        target_col = int(random_generator.integers(0, n_cols))
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
                               objective_state: ObjectiveStateND) -> Dict[str, object]:
    return objective_function(
        candidate_matrix=candidate_matrix,
        objective_state=objective_state,
    )


def _apply_perturbation(candidate_matrix: np.ndarray,
                        random_generator: np.random.Generator,
                        perturbation_strength: int,
                        allowed_source_rows: np.ndarray) -> int:
    applied_moves = 0
    for _ in range(max(0, int(perturbation_strength))):
        relocation_move = _sample_feasible_relocation_move(
            candidate_matrix=candidate_matrix,
            random_generator=random_generator,
            allowed_source_rows=allowed_source_rows,
        )
        if relocation_move is None:
            break
        source_row, source_col, target_row, target_col = relocation_move
        _apply_relocation_move(candidate_matrix, source_row, source_col, target_row, target_col)
        applied_moves += 1
    return applied_moves


def _run_local_search_first_improvement(candidate_matrix: np.ndarray,
                                        objective_state: ObjectiveStateND,
                                        random_generator: np.random.Generator,
                                        current_objective_value: float,
                                        max_local_search_steps: int,
                                        local_search_neighbor_sample: int,
                                        allowed_source_rows: np.ndarray,
                                        objective_epsilon: float) -> Dict[str, object]:
    total_evaluations = 0
    accepted_local_moves = 0
    local_search_steps = 0
    objective_value = float(current_objective_value)

    for _ in range(max_local_search_steps):
        local_search_steps += 1
        found_improvement = False
        for _ in range(local_search_neighbor_sample):
            relocation_move = _sample_feasible_relocation_move(
                candidate_matrix=candidate_matrix,
                random_generator=random_generator,
                allowed_source_rows=allowed_source_rows,
            )
            if relocation_move is None:
                break

            source_row, source_col, target_row, target_col = relocation_move
            _apply_relocation_move(candidate_matrix, source_row, source_col, target_row, target_col)

            evaluation_result = _evaluate_candidate_matrix(candidate_matrix, objective_state)
            total_evaluations += 1
            trial_objective_value = float(evaluation_result['objective_value'])

            if trial_objective_value > (objective_value + objective_epsilon):
                objective_value = trial_objective_value
                accepted_local_moves += 1
                found_improvement = True
                break

            _apply_relocation_move(candidate_matrix, target_row, target_col, source_row, source_col)

        if not found_improvement:
            break

    return {
        "objective_value": objective_value,
        "total_evaluations": total_evaluations,
        "accepted_local_moves": accepted_local_moves,
        "local_search_steps": local_search_steps,
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
        }

    objective_values = [float(item["best_objective_value"]) for item in per_seed_run_summaries]
    runtime_values = [float(item["runtime_seconds"]) for item in per_seed_run_summaries]
    evaluation_values = [int(item["evaluations"]) for item in per_seed_run_summaries]

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
    initial_eval_result = _evaluate_candidate_matrix(current_candidate_matrix, objective_state)
    initial_objective_value = float(initial_eval_result["objective_value"])
    evaluations = 1
    iterations = 0

    initial_local_search_result = _run_local_search_first_improvement(
        candidate_matrix=current_candidate_matrix,
        objective_state=objective_state,
        random_generator=random_generator,
        current_objective_value=initial_objective_value,
        max_local_search_steps=config.local_search_max_steps,
        local_search_neighbor_sample=config.local_search_neighbor_sample,
        allowed_source_rows=allowed_source_rows,
        objective_epsilon=config.objective_epsilon,
    )
    current_objective_value = float(initial_local_search_result["objective_value"])
    evaluations += int(initial_local_search_result["total_evaluations"])

    best_candidate_matrix = current_candidate_matrix.copy()
    best_objective_value = current_objective_value

    accepted_iterations = 0
    improvement_iterations = 0
    no_improve_streak = 0

    perturb_current = max(1, min(int(config.perturbation_strength), int(context.budget)))
    progress_log_every = max(1, int(config.log_every_iterations))
    if config.log_enabled:
        print(
            f"[ILS][seed={seed_value}] start "
            f"initial={initial_objective_value:.4f} "
            f"after_ls={current_objective_value:.4f} "
            f"best={best_objective_value:.4f} "
            f"perturbation={perturb_current} "
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
    }]

    while evaluations < config.max_evaluations and iterations < config.max_iterations and no_improve_streak < config.max_no_improve:
        iterations += 1
        best_objective_before_iteration = best_objective_value

        candidate_matrix = current_candidate_matrix.copy()
        applied_perturbation_moves = _apply_perturbation(
            candidate_matrix=candidate_matrix,
            random_generator=random_generator,
            perturbation_strength=perturb_current,
            allowed_source_rows=allowed_source_rows,
        )

        candidate_eval_result = _evaluate_candidate_matrix(candidate_matrix, objective_state)
        candidate_objective_value = float(candidate_eval_result["objective_value"])
        evaluations += 1

        local_search_result = _run_local_search_first_improvement(
            candidate_matrix=candidate_matrix,
            objective_state=objective_state,
            random_generator=random_generator,
            current_objective_value=candidate_objective_value,
            max_local_search_steps=config.local_search_max_steps,
            local_search_neighbor_sample=config.local_search_neighbor_sample,
            allowed_source_rows=allowed_source_rows,
            objective_epsilon=config.objective_epsilon,
        )
        candidate_objective_value = float(local_search_result["objective_value"])
        evaluations += int(local_search_result["total_evaluations"])

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
            "perturbation_strength": int(perturb_current),
            "no_improve_streak": int(no_improve_streak),
            "local_search_accepted_moves": int(local_search_result["accepted_local_moves"]),
            "applied_perturbation_moves": int(applied_perturbation_moves),
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
                    or (config.log_on_acceptance and accepted)
                )

        if should_log_progress:
            print(
                f"[ILS][seed={seed_value}] iter={iterations} eval={evaluations} "
                f"candidate={candidate_objective_value:.4f} current={current_objective_value:.4f} "
                f"best={best_objective_value:.4f} perturbation={perturb_current} "
                f"pert_moves={applied_perturbation_moves} accepted={accepted} "
                f"improved={improved} no_improve={no_improve_streak} "
                f"ls_acc={int(local_search_result['accepted_local_moves'])}"
            )

    runtime_seconds = float(perf_counter() - start_time)
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
            f"stop={stopping_reason}"
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
    }


def run_ils(context: MetaheuristicContext) -> dict:
    """Run full Iterated Local Search experiment across all seeds."""
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

    config = ILSRuntimeConfig(
        perturbation_strength=max(1, int(os.getenv("ILS_PERTURBATION_STRENGTH", "2"))),
        acceptance_criterion="better",
        experiment_mode=os.getenv("ILS_EXPERIMENT_MODE", "1") != "0",
        debug_mode=os.getenv("ILS_DEBUG_MODE", "0") == "1",
        save_best_matrix_npz=os.getenv("ILS_SAVE_BEST_MATRIX_NPZ", "0") == "1",
        log_enabled=os.getenv("ILS_LOG_ENABLED", "1") == "1",
        log_every_iterations=max(1, int(os.getenv("ILS_LOG_EVERY", "10"))),
        log_only_improvements=os.getenv("ILS_LOG_ONLY_IMPROVEMENTS", "0") == "1",
        log_on_acceptance=os.getenv("ILS_LOG_ON_ACCEPTANCE", "0") == "1",
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

    for seed in context.seeds:
        seed_value = int(seed)
        initial_candidate_matrix = seed_to_initial_candidate[seed_value].copy()

        debug_files = {}
        if config.debug_mode:
            debug_files = save_nd_debug_matrices(
                context=context,
                seed=seed_value,
                candidate_matrix=initial_candidate_matrix,
            )

        seed_result = _run_single_seed_ils(
            seed_value=seed_value,
            context=context,
            objective_state=objective_state,
            initial_candidate_matrix=initial_candidate_matrix,
            config=config,
        )
        seed_artifact_paths = {}
        if config.experiment_mode and experiment_directories is not None:
            seed_artifact_paths = _save_per_seed_artifacts(
                experiment_directories=experiment_directories,
                per_seed_result=seed_result,
                objective_state=objective_state,
                save_best_matrix_npz=config.save_best_matrix_npz,
            )
        seed_result["artifact_paths"] = seed_artifact_paths
        if debug_files:
            seed_result["debug_files"] = debug_files
        per_seed_result_records.append(seed_result)

        per_seed_run_summaries.append({
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
        })

    global_best_seed_result = max(
        per_seed_result_records,
        key=lambda item: float(item["best_objective_value"]),
    )
    summary_stats = _compute_summary_statistics(per_seed_run_summaries)

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
        "global_best_seed": int(global_best_seed_result["seed"]),
        "global_best_objective_value": float(global_best_seed_result["best_objective_value"]),
        "global_best_allocation_size": len(global_best_seed_result["best_allocation_items"]),
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
