from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObjectiveStateND:
    """Precompiled numeric state for low-overhead candidate evaluation."""
    h3_ids: List[str]
    h3_to_index: Dict[str, int]
    candidate_dimensions: List[str]
    dimension_to_index: Dict[str, int]
    indicator_columns: List[str]
    candidate_to_indicator_indices: np.ndarray
    baseline_matrix: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    alpha_values: np.ndarray

    @property
    def base_indicator_matrix(self) -> np.ndarray:
        """Backward-compatible alias for baseline_matrix."""
        return self.baseline_matrix


@dataclass
class MetaheuristicContext:
    """Shared context passed to each metaheuristic method implementation."""
    df_walkability: pd.DataFrame
    df_hex_time_matrix: pd.DataFrame
    location: str
    budget: int
    method_code: str
    method_name: str
    seeds: List[int]
    walking_profile: str
    dimensions: List[str]
    source_hex_ids: List[str]
    baseline_iqc_total: Optional[float]
    allocations: List[Dict[str, object]]
    objective_state_nd: Optional[ObjectiveStateND] = None
