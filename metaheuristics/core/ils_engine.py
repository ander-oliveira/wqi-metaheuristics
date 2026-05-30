"""
Compatibility layer for older imports that referenced `metaheuristics.core.ils_engine`.

Current canonical implementation lives in:
- `metaheuristics.methods.ils.run_ils`
- `metaheuristics.core.evaluation` (ndarray objective helpers)
"""

from dataclasses import dataclass

from .evaluation import (
    allocation_items_to_candidate_matrix,
    build_objective_state_nd,
    evaluate_candidate_matrix_nd,
    objective_function,
    objective_function_with_time_nd,
)
from .types import MetaheuristicContext, ObjectiveStateND


@dataclass(frozen=True)
class ILSRuntimeConfig:
    """Backward-compatible config signature."""
    max_evaluations: int = 30000
    max_iterations: int = 500
    max_no_improve: int = 80
    local_search_neighbor_sample: int = 64
    local_search_max_steps: int = 25
    perturbation_strength_ratio_initial: float = 0.05
    perturbation_strength_ratio_maximum: float = 0.25
    perturbation_strength_minimum: int = 1
    objective_epsilon: float = 1e-12
    acceptance_criterion: str = "better"
    experiment_mode: bool = True
    debug_mode: bool = False
    save_best_matrix_npz: bool = False


def run_ils_experiment(context: MetaheuristicContext):
    """Compatibility wrapper that delegates to the method runner implementation."""
    from ..methods.ils import run_ils  # Local import avoids circular dependency during package init.
    return run_ils(context)


def run_ils_multi_seed_experiment(context: MetaheuristicContext):
    """Alias kept for compatibility with previous naming."""
    return run_ils_experiment(context)


__all__ = [
    "ILSRuntimeConfig",
    "MetaheuristicContext",
    "ObjectiveStateND",
    "allocation_items_to_candidate_matrix",
    "build_objective_state_nd",
    "evaluate_candidate_matrix_nd",
    "objective_function",
    "objective_function_with_time_nd",
    "run_ils_experiment",
    "run_ils_multi_seed_experiment",
]
