from .budget import (
    POI_DIMENSION_COLUMNS,
    generate_random_allocations,
    generate_random_spatial_allocations,
    random_budget_allocation,
    random_spatial_budget_allocation,
)
from .evaluation import (
    CORE_INDICATOR_COLUMNS,
    HEX_TIME_MATRIX_REQUIRED_COLUMNS,
    ID_COLUMNS,
    apply_spatial_allocation_with_time,
    allocation_items_to_candidate_matrix,
    build_objective_state_nd,
    calculate_time_decay_weight,
    compute_baseline_iqc_total,
    evaluate_candidate_matrix_nd,
    get_available_dimensions,
    objective_function,
    objective_function_with_time_nd,
    recalculate_iqc_and_critic,
    validate_hex_time_matrix,
)
from .io import load_seeds
from .types import MetaheuristicContext, ObjectiveStateND

try:
    from .ils_engine import (
        ILSRuntimeConfig,
        run_ils_experiment,
        run_ils_multi_seed_experiment,
    )
except Exception:  # pragma: no cover - compatibility path
    ILSRuntimeConfig = None
    run_ils_experiment = None
    run_ils_multi_seed_experiment = None

__all__ = [
    'CORE_INDICATOR_COLUMNS',
    'HEX_TIME_MATRIX_REQUIRED_COLUMNS',
    'ID_COLUMNS',
    'ILSRuntimeConfig',
    'ObjectiveStateND',
    'POI_DIMENSION_COLUMNS',
    'MetaheuristicContext',
    'apply_spatial_allocation_with_time',
    'allocation_items_to_candidate_matrix',
    'build_objective_state_nd',
    'calculate_time_decay_weight',
    'compute_baseline_iqc_total',
    'evaluate_candidate_matrix_nd',
    'generate_random_allocations',
    'generate_random_spatial_allocations',
    'get_available_dimensions',
    'load_seeds',
    'objective_function',
    'objective_function_with_time_nd',
    'recalculate_iqc_and_critic',
    'run_ils_experiment',
    'run_ils_multi_seed_experiment',
    'random_budget_allocation',
    'random_spatial_budget_allocation',
    'validate_hex_time_matrix',
]
