import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from walkability.iqc import calculate_walkability_index
from .types import ObjectiveStateND


ID_COLUMNS = ['h3_id', 'latitude', 'longitude']
CORE_INDICATOR_COLUMNS = [
    'S_saude',
    'S_educacao',
    'S_abastecimento',
    'S_lazer',
    'S_servicos',
    'I_seguranca',
    'A_vegetacao',
    'A_agua',
    'C_conectividade',
    'T_transporte',
    'U_urbanidade',
]

HEX_TIME_MATRIX_REQUIRED_COLUMNS = ['source_h3_id', 'target_h3_id', 'time_min']


def get_available_dimensions(df_walkability: pd.DataFrame,
                             candidate_dimensions: Iterable[str]) -> List[str]:
    """Return dimensions that exist in the provided walkability dataframe."""
    if df_walkability is None or df_walkability.empty:
        return []
    return [col for col in candidate_dimensions if col in df_walkability.columns]


def compute_baseline_iqc_total(df_walkability: pd.DataFrame) -> Optional[float]:
    """Compute baseline global IQC (sum of IQC) when IQC column exists."""
    if df_walkability is None or df_walkability.empty:
        return None
    if 'IQC' not in df_walkability.columns:
        return None
    return float(df_walkability['IQC'].sum())


def calculate_time_decay_weight(time_min: float, max_time: float = 20.0) -> float:
    """Cosine-decay accessibility weight."""
    if pd.isna(time_min):
        return 0.0
    t = float(time_min)
    if t < 0 or t > max_time:
        return 0.0
    return float((1 + math.cos(math.pi * t / max_time)) / 2)


def validate_hex_time_matrix(df_hex_time_matrix: pd.DataFrame,
                             max_time: float = 20.0) -> pd.DataFrame:
    """
    Validate and normalize hex impact matrix.

    Required columns: source_h3_id, target_h3_id, time_min.
    If alpha_20 is absent, it is computed from time_min.
    """
    if df_hex_time_matrix is None or df_hex_time_matrix.empty:
        raise ValueError("df_hex_time_matrix is empty.")

    missing_cols = [col for col in HEX_TIME_MATRIX_REQUIRED_COLUMNS if col not in df_hex_time_matrix.columns]
    if missing_cols:
        raise ValueError(f"df_hex_time_matrix is missing required columns: {missing_cols}")

    df_matrix = df_hex_time_matrix.copy()
    df_matrix['source_h3_id'] = df_matrix['source_h3_id'].astype(str)
    df_matrix['target_h3_id'] = df_matrix['target_h3_id'].astype(str)
    df_matrix['time_min'] = pd.to_numeric(df_matrix['time_min'], errors='coerce')
    df_matrix = df_matrix.dropna(subset=['source_h3_id', 'target_h3_id', 'time_min'])
    df_matrix = df_matrix[df_matrix['time_min'] <= max_time].copy()
    if df_matrix.empty:
        raise ValueError("df_hex_time_matrix has no rows with time_min <= max_time.")

    if 'alpha_20' in df_matrix.columns:
        df_matrix['alpha_20'] = pd.to_numeric(df_matrix['alpha_20'], errors='coerce')
    else:
        df_matrix['alpha_20'] = df_matrix['time_min'].apply(
            lambda t: calculate_time_decay_weight(t, max_time=max_time)
        )

    df_matrix['alpha_20'] = df_matrix['alpha_20'].fillna(0.0)
    df_matrix = df_matrix[df_matrix['alpha_20'] > 0].copy()
    if df_matrix.empty:
        raise ValueError("df_hex_time_matrix has no positive alpha_20 rows.")

    # Keep one row per pair (best/shortest path).
    df_matrix = (
        df_matrix.sort_values(['source_h3_id', 'target_h3_id', 'time_min'])
        .drop_duplicates(subset=['source_h3_id', 'target_h3_id'], keep='first')
        .reset_index(drop=True)
    )
    return df_matrix


def normalize_spatial_allocation(allocation_items: Iterable[Dict[str, object]],
                                 valid_dimensions: Iterable[str]) -> pd.DataFrame:
    """Normalize allocation items to a compact dataframe."""
    if allocation_items is None:
        return pd.DataFrame(columns=['h3_id', 'dimension', 'quantity'])

    valid_dimensions = set(valid_dimensions)
    df_alloc = pd.DataFrame(list(allocation_items))
    if df_alloc.empty:
        return pd.DataFrame(columns=['h3_id', 'dimension', 'quantity'])

    required_cols = ['h3_id', 'dimension', 'quantity']
    missing_cols = [col for col in required_cols if col not in df_alloc.columns]
    if missing_cols:
        raise ValueError(f"Allocation items are missing required fields: {missing_cols}")

    df_alloc['h3_id'] = df_alloc['h3_id'].astype(str)
    df_alloc['dimension'] = df_alloc['dimension'].astype(str)
    df_alloc['quantity'] = pd.to_numeric(df_alloc['quantity'], errors='coerce').fillna(0.0)
    df_alloc = df_alloc[df_alloc['quantity'] > 0].copy()
    df_alloc = df_alloc[df_alloc['dimension'].isin(valid_dimensions)].copy()
    if df_alloc.empty:
        return pd.DataFrame(columns=['h3_id', 'dimension', 'quantity'])

    df_alloc = (
        df_alloc.groupby(['h3_id', 'dimension'], as_index=False)['quantity']
        .sum()
        .sort_values(['h3_id', 'dimension'])
        .reset_index(drop=True)
    )
    return df_alloc


def apply_spatial_allocation_with_time(df_walkability: pd.DataFrame,
                                       df_hex_time_matrix: pd.DataFrame,
                                       allocation_items: Iterable[Dict[str, object]],
                                       candidate_dimensions: Iterable[str],
                                       max_time: float = 20.0) -> pd.DataFrame:
    """
    Apply a spatial POI allocation using source-target alpha impact matrix.
    """
    if df_walkability is None or df_walkability.empty:
        raise ValueError("df_walkability is empty.")

    if 'h3_id' not in df_walkability.columns:
        raise ValueError("df_walkability is missing required column 'h3_id'.")

    available_dimensions = get_available_dimensions(df_walkability, candidate_dimensions)
    if not available_dimensions:
        raise ValueError("No candidate POI dimensions found in df_walkability.")

    df_matrix = validate_hex_time_matrix(df_hex_time_matrix, max_time=max_time)
    df_alloc = normalize_spatial_allocation(allocation_items, available_dimensions)

    df_updated = df_walkability.copy()
    df_updated['h3_id'] = df_updated['h3_id'].astype(str)

    # Ensure indicator columns are numeric before updates.
    for dim in available_dimensions:
        df_updated[dim] = pd.to_numeric(df_updated[dim], errors='coerce').fillna(0.0)

    if df_alloc.empty:
        return df_updated

    # Keep only source hexagons represented in the matrix.
    df_alloc = df_alloc[df_alloc['h3_id'].isin(df_matrix['source_h3_id'].unique())].copy()
    if df_alloc.empty:
        return df_updated

    merged = df_alloc.merge(
        df_matrix[['source_h3_id', 'target_h3_id', 'alpha_20']],
        left_on='h3_id',
        right_on='source_h3_id',
        how='inner',
    )
    if merged.empty:
        return df_updated

    merged['delta'] = merged['quantity'] * merged['alpha_20']
    grouped = (
        merged.groupby(['target_h3_id', 'dimension'], as_index=False)['delta']
        .sum()
    )
    if grouped.empty:
        return df_updated

    delta_pivot = grouped.pivot(index='target_h3_id', columns='dimension', values='delta').fillna(0.0)
    for dim in delta_pivot.columns:
        dim_delta = delta_pivot[dim]
        df_updated[dim] = df_updated[dim] + df_updated['h3_id'].map(dim_delta).fillna(0.0)

    return df_updated


def recalculate_iqc_and_critic(df_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Recalculate CRITIC weights and IQC from an updated dataframe.

    Expected input is the current solution dataframe (`df_final`), usually based on
    `df_walkability` after new POI allocations. Existing IQC is discarded and
    recomputed for all hexagons.
    """
    if df_final is None or df_final.empty:
        raise ValueError("df_final is empty.")

    missing_id_cols = [col for col in ID_COLUMNS if col not in df_final.columns]
    if missing_id_cols:
        raise ValueError(f"df_final is missing required ID columns: {missing_id_cols}")

    available_indicator_cols = [col for col in CORE_INDICATOR_COLUMNS if col in df_final.columns]
    if not available_indicator_cols:
        raise ValueError("df_final has no core indicator columns for IQC recalculation.")

    df_indicators = df_final[ID_COLUMNS + available_indicator_cols].copy()
    if 'IQC' in df_indicators.columns:
        df_indicators = df_indicators.drop(columns=['IQC'])

    for col in available_indicator_cols:
        df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')
    df_indicators[available_indicator_cols] = df_indicators[available_indicator_cols].fillna(0.0)

    df_recomputed, critic_weights = calculate_walkability_index(df_indicators)

    df_updated = df_final.copy()
    df_updated['IQC'] = df_recomputed['IQC'].values

    return df_updated, critic_weights


def objective_function(*,
                       final_indicator_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Official objective function entry point.

    This function only evaluates a ready indicator matrix:
    - recalculates CRITIC weights,
    - recalculates IQC,
    - returns sum(IQC) as objective value.
    """
    indicator_matrix = np.asarray(final_indicator_matrix, dtype=np.float64)
    if indicator_matrix.ndim != 2:
        raise ValueError("final_indicator_matrix must be a 2D ndarray.")
    if indicator_matrix.shape[0] == 0 or indicator_matrix.shape[1] == 0:
        raise ValueError("final_indicator_matrix must have at least one row and one column.")

    critic_weights = _compute_critic_weights_numpy(indicator_matrix)
    iqc_values = _compute_iqc_numpy(indicator_matrix, critic_weights=critic_weights)
    objective_value = float(iqc_values.sum())

    return {
        'objective_metric': 'sum_iqc',
        'objective_value': objective_value,
        'optimization_direction': 'maximize',
        'critic_weights': critic_weights,
        'iqc_values': iqc_values,
    }


def build_objective_state_nd(df_walkability: pd.DataFrame,
                             df_hex_time_matrix: pd.DataFrame,
                             candidate_dimensions: Iterable[str],
                             max_time: float = 20.0) -> ObjectiveStateND:
    """
    Precompile numeric structures used by the ndarray objective function.

    This should run once per optimization execution. Candidate evaluation
    should then use:
    1) `build_final_indicator_matrix_nd(candidate_matrix=..., objective_state=...)`
    2) `objective_function(final_indicator_matrix=...)`
    """
    if df_walkability is None or df_walkability.empty:
        raise ValueError("df_walkability is empty.")
    if 'h3_id' not in df_walkability.columns:
        raise ValueError("df_walkability is missing required column 'h3_id'.")

    df_base = df_walkability.copy()
    df_base['h3_id'] = df_base['h3_id'].astype(str)
    if df_base['h3_id'].duplicated().any():
        df_base = df_base.drop_duplicates(subset=['h3_id'], keep='first').reset_index(drop=True)

    indicator_columns = [col for col in CORE_INDICATOR_COLUMNS if col in df_base.columns]
    if not indicator_columns:
        raise ValueError("df_walkability has no core indicator columns for ndarray objective.")

    for col in indicator_columns:
        df_base[col] = pd.to_numeric(df_base[col], errors='coerce')
    df_base[indicator_columns] = df_base[indicator_columns].fillna(0.0)

    candidate_dims = [str(dim) for dim in candidate_dimensions if str(dim) in indicator_columns]
    if not candidate_dims:
        raise ValueError("No candidate POI dimensions found for ndarray objective.")

    h3_ids = df_base['h3_id'].tolist()
    h3_to_index = {h3_id: idx for idx, h3_id in enumerate(h3_ids)}

    candidate_to_indicator_indices = np.asarray(
        [indicator_columns.index(dim) for dim in candidate_dims],
        dtype=np.int32,
    )
    dimension_to_index = {dim: idx for idx, dim in enumerate(candidate_dims)}

    baseline_matrix = df_base[indicator_columns].to_numpy(dtype=np.float64, copy=True)

    df_matrix = validate_hex_time_matrix(df_hex_time_matrix, max_time=max_time)

    source_hex = df_matrix['source_h3_id'].astype(str).to_numpy()
    target_hex = df_matrix['target_h3_id'].astype(str).to_numpy()
    alpha_values_raw = pd.to_numeric(df_matrix['alpha_20'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)

    valid_rows_mask = np.asarray(
        [(src in h3_to_index) and (tgt in h3_to_index) for src, tgt in zip(source_hex, target_hex)],
        dtype=bool,
    )
    if not valid_rows_mask.any():
        raise ValueError("No source-target rows in hex-time matrix map to df_walkability h3_id values.")

    source_indices = np.asarray([h3_to_index[src] for src in source_hex[valid_rows_mask]], dtype=np.int32)
    target_indices = np.asarray([h3_to_index[tgt] for tgt in target_hex[valid_rows_mask]], dtype=np.int32)
    alpha_values = alpha_values_raw[valid_rows_mask].astype(np.float64, copy=False)

    return ObjectiveStateND(
        h3_ids=h3_ids,
        h3_to_index=h3_to_index,
        candidate_dimensions=candidate_dims,
        dimension_to_index=dimension_to_index,
        indicator_columns=indicator_columns,
        candidate_to_indicator_indices=candidate_to_indicator_indices,
        baseline_matrix=baseline_matrix,
        source_indices=source_indices,
        target_indices=target_indices,
        alpha_values=alpha_values,
    )


def allocation_items_to_candidate_matrix(allocation_items: Iterable[Dict[str, object]],
                                         objective_state: ObjectiveStateND) -> np.ndarray:
    """
    Convert allocation items into candidate matrix.

    Output matrix shape is `(n_hex, n_candidate_dimensions)` where each row is
    the internal sequential index of the hexagon.
    """
    n_hex = len(objective_state.h3_ids)
    n_dims = len(objective_state.candidate_dimensions)
    candidate_matrix = np.zeros((n_hex, n_dims), dtype=np.float64)

    if allocation_items is None:
        return candidate_matrix

    for item in allocation_items:
        if not isinstance(item, dict):
            raise ValueError("Each allocation item must be a dictionary.")
        missing_fields = [field for field in ['h3_id', 'dimension', 'quantity'] if field not in item]
        if missing_fields:
            raise ValueError(f"Allocation item is missing required fields: {missing_fields}")

        h3_id = str(item['h3_id'])
        dimension = str(item['dimension'])
        quantity_raw = item['quantity']

        if h3_id not in objective_state.h3_to_index:
            continue
        if dimension not in objective_state.dimension_to_index:
            continue

        try:
            quantity = float(quantity_raw)
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue

        row_idx = objective_state.h3_to_index[h3_id]
        col_idx = objective_state.dimension_to_index[dimension]
        candidate_matrix[row_idx, col_idx] += quantity

    return candidate_matrix


def _compute_critic_weights_numpy(indicator_matrix: np.ndarray) -> np.ndarray:
    """Numpy implementation of CRITIC weights compatible with current pipeline."""
    if indicator_matrix.ndim != 2:
        raise ValueError("indicator_matrix must be a 2D array.")

    n_rows, n_cols = indicator_matrix.shape
    if n_cols == 0:
        raise ValueError("indicator_matrix has zero columns.")
    if n_rows == 0:
        raise ValueError("indicator_matrix has zero rows.")

    col_min = indicator_matrix.min(axis=0)
    col_max = indicator_matrix.max(axis=0)
    col_range = col_max - col_min

    df_norm = np.zeros_like(indicator_matrix, dtype=np.float64)
    non_zero_range = col_range > 0
    if non_zero_range.any():
        df_norm[:, non_zero_range] = (
            (indicator_matrix[:, non_zero_range] - col_min[non_zero_range]) / col_range[non_zero_range]
        )

    ddof = 1 if n_rows > 1 else 0
    std_dev = df_norm.std(axis=0, ddof=ddof)

    valid_cols = np.where(non_zero_range)[0]
    if valid_cols.size < 2:
        return np.full(n_cols, 1.0 / float(n_cols), dtype=np.float64)

    corr_matrix = np.corrcoef(indicator_matrix[:, valid_cols], rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    corr_abs = np.abs(corr_matrix)

    conflict = np.zeros(n_cols, dtype=np.float64)
    conflict[valid_cols] = (1.0 - corr_abs).sum(axis=1)

    critic_measure = std_dev * conflict
    critic_sum = float(critic_measure.sum())
    if critic_sum > 0.0:
        return critic_measure / critic_sum

    return np.full(n_cols, 1.0 / float(n_cols), dtype=np.float64)


def _compute_iqc_numpy(indicator_matrix: np.ndarray,
                       critic_weights: np.ndarray) -> np.ndarray:
    """Numpy implementation of IQC aggregation (with rounding to 4 decimals)."""
    if indicator_matrix.ndim != 2:
        raise ValueError("indicator_matrix must be a 2D array.")
    if critic_weights.ndim != 1:
        raise ValueError("critic_weights must be a 1D array.")
    if indicator_matrix.shape[1] != critic_weights.shape[0]:
        raise ValueError("critic_weights size does not match indicator_matrix columns.")

    col_min = indicator_matrix.min(axis=0)
    col_max = indicator_matrix.max(axis=0)
    col_range = col_max - col_min

    df_norm = np.full_like(indicator_matrix, 0.5, dtype=np.float64)
    non_zero_range = col_range > 0
    if non_zero_range.any():
        df_norm[:, non_zero_range] = (
            (indicator_matrix[:, non_zero_range] - col_min[non_zero_range]) / col_range[non_zero_range]
        )

    iqc_values = df_norm @ critic_weights
    return np.round(iqc_values, 4)


def build_final_indicator_matrix_nd(candidate_matrix: np.ndarray,
                                    objective_state: ObjectiveStateND) -> np.ndarray:
    """
    Build final indicator matrix from baseline + propagated candidate impact.

    This function does not evaluate CRITIC/IQC; it only applies the
    source-target alpha propagation and returns the updated indicator matrix.
    """
    candidate_array = np.asarray(candidate_matrix, dtype=np.float64)

    n_hex = len(objective_state.h3_ids)
    n_dims = len(objective_state.candidate_dimensions)
    expected_shape = (n_hex, n_dims)

    if candidate_array.ndim != 2:
        raise ValueError("candidate_matrix must be a 2D ndarray.")
    if candidate_array.shape != expected_shape:
        raise ValueError(
            f"candidate_matrix shape {candidate_array.shape} does not match expected {expected_shape}."
        )

    allocation_rows = candidate_array[objective_state.source_indices]
    weighted_rows = allocation_rows * objective_state.alpha_values[:, None]

    delta_by_target = np.zeros((n_hex, n_dims), dtype=np.float64)
    np.add.at(delta_by_target, objective_state.target_indices, weighted_rows)

    final_indicator_matrix = objective_state.baseline_matrix.copy()
    final_indicator_matrix[:, objective_state.candidate_to_indicator_indices] += delta_by_target
    return final_indicator_matrix


def evaluate_candidate_matrix_nd(candidate_matrix: np.ndarray,
                                 objective_state: ObjectiveStateND) -> Dict[str, object]:
    """
    Compatibility wrapper:
    candidate_matrix -> final_indicator_matrix -> objective_function.
    """
    candidate_array = np.asarray(candidate_matrix, dtype=np.float64)
    final_indicator_matrix = build_final_indicator_matrix_nd(
        candidate_matrix=candidate_array,
        objective_state=objective_state,
    )
    eval_result = objective_function(final_indicator_matrix=final_indicator_matrix)
    eval_result['applied_allocation_size'] = int(np.count_nonzero(candidate_array))
    return eval_result
