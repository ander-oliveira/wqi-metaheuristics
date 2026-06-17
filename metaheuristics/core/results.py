"""Persistence of metaheuristic runs for statistical analysis and reproducibility.

Every optimization run produces:

1. A per-run directory ``data/optimization_runs/<location>/<profile>/<method>/<ts>__<id>/``
   containing the full detail of that single run:
     - run_metadata.json          full reproducible metadata (params, env, git, ...)
     - run_report.txt             human-readable summary
     - run_summary.csv            one-row summary (this run)
     - per_seed.csv               one row per seed (stochastic variability)
     - per_hexagon_iqc.csv        baseline vs optimized IQC per hexagon (+ delta)
     - per_hexagon_indicators.csv tidy indicator values baseline vs optimized
     - allocation.csv             best allocation (hexagon, dimension, quantity)
     - critic_weights.csv         CRITIC weights baseline vs optimized
     - convergence.csv            objective per construction step, per seed

2. Append-only master tables under ``data/optimization_runs/`` so every run from
   every location/profile/method accumulates into a single tidy file ready for
   cross-run statistics:
     - all_runs_summary.csv       one row per run
     - all_per_seed.csv           one row per (run, seed)
     - all_per_hexagon_iqc.csv    one row per (run, hexagon)
     - all_allocations.csv        one row per (run, allocation item)

The schema of each master table is fixed (see the ``*_COLUMNS`` lists) so that
files concatenate cleanly across runs even as methods evolve.
"""

import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from walkability.utils import sanitize_path_component


OPTIMIZATION_RUNS_ROOT = os.path.join('data', 'optimization_runs')

RUN_SUMMARY_COLUMNS = [
    'run_id', 'timestamp_utc',
    'location', 'key_location', 'walking_profile',
    'method_code', 'method_name', 'status',
    'budget', 'n_seeds', 'best_seed',
    'objective_metric', 'optimization_direction',
    'baseline_objective', 'best_objective', 'improvement', 'improvement_pct',
    'obj_seed_mean', 'obj_seed_std', 'obj_seed_min', 'obj_seed_max',
    'iqc_base_mean', 'iqc_base_std',
    'iqc_opt_mean', 'iqc_opt_std', 'iqc_opt_min', 'iqc_opt_median', 'iqc_opt_max',
    'delta_iqc_mean', 'delta_iqc_std', 'delta_iqc_min', 'delta_iqc_max',
    'delta_iqc_positive_count',
    'n_hexagons', 'n_source_hexagons', 'n_candidate_dimensions',
    'allocated_pois', 'distinct_pairs',
    'runtime_seconds_total',
    'rcl_alpha', 'construction_sample_size', 'local_search_max_evals',
    'improvement_eps', 'parallel_workers', 'eval_cache',
    'h3_resolution', 'distance', 'run_dir',
]

PER_SEED_COLUMNS = [
    'run_id', 'location', 'key_location', 'walking_profile', 'method_code',
    'seed', 'construction_objective', 'local_search_objective',
    'local_search_gain', 'improvement_over_baseline', 'runtime_seconds',
    'distinct_pairs',
]

PER_HEXAGON_COLUMNS = [
    'run_id', 'location', 'key_location', 'walking_profile', 'method_code',
    'h3_id', 'latitude', 'longitude',
    'iqc_baseline', 'iqc_optimized', 'delta_iqc', 'iqc_pipeline',
]

ALLOCATION_COLUMNS = [
    'run_id', 'location', 'key_location', 'walking_profile', 'method_code',
    'h3_id', 'dimension', 'quantity', 'latitude', 'longitude',
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _append_master(path: str, df: pd.DataFrame, columns: List[str]) -> None:
    """Append rows to a master CSV with a fixed column order (header once)."""
    df = df.reindex(columns=columns)
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False, encoding='utf-8')


def _safe(stats: dict, key, default=None):
    value = stats.get(key, default)
    return value


def _summ(arr) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        'mean': float(np.mean(a)), 'std': float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        'min': float(np.min(a)), 'max': float(np.max(a)), 'median': float(np.median(a)),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def persist_run(*,
                context,
                method_result: dict,
                location: Optional[str] = None,
                key_location: Optional[str] = None,
                h3_resolution: Optional[int] = None,
                distance: Optional[int] = None,
                input_files: Optional[Dict[str, str]] = None,
                runs_root: str = OPTIMIZATION_RUNS_ROOT) -> dict:
    """Persist a full optimization run. Returns a dict of written file paths.

    Designed to be method-agnostic: it writes whatever data is available in
    ``method_result`` (rich for GRASP; minimal for placeholder methods) and never
    raises -- persistence failures must not abort an optimization.
    """
    paths: Dict[str, str] = {}
    try:
        run_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc)
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        profile = getattr(context, 'walking_profile', None) or 'unknown_profile'
        location = location or 'unknown_location'
        key_location = key_location or location
        method_code = method_result.get('method_code', getattr(context, 'method_code', '?'))
        method_name = method_result.get('method_name', getattr(context, 'method_name', '?'))
        status = method_result.get('status', 'unknown')

        run_dir = os.path.join(
            runs_root,
            sanitize_path_component(key_location),
            sanitize_path_component(profile),
            sanitize_path_component(method_code),
            f"{timestamp}__{run_id}",
        )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(runs_root, exist_ok=True)

        stats = method_result.get('statistics', {}) or {}
        summary = method_result.get('best_solution_summary', {}) or {}

        # ---- scalar fields -------------------------------------------------
        baseline_obj = stats.get('baseline_objective', summary.get('baseline_objective'))
        best_obj = method_result.get('best_objective_value')
        improvement = (best_obj - baseline_obj) if (best_obj is not None and baseline_obj is not None) else None
        improvement_pct = (
            100.0 * improvement / baseline_obj
            if (improvement is not None and baseline_obj not in (None, 0)) else None
        )

        per_seed = stats.get('per_seed', []) or []
        ls_objectives = [s.get('local_search_objective') for s in per_seed
                         if s.get('local_search_objective') is not None]
        seed_summ = _summ(ls_objectives) if ls_objectives else {}

        iqc_base = np.asarray(stats.get('iqc_baseline', []), dtype=np.float64)
        iqc_opt = np.asarray(stats.get('iqc_optimized', []), dtype=np.float64)
        delta_iqc = (iqc_opt - iqc_base) if (iqc_base.size and iqc_base.size == iqc_opt.size) else np.array([])
        base_summ = _summ(iqc_base) if iqc_base.size else {}
        opt_summ = _summ(iqc_opt) if iqc_opt.size else {}
        delta_summ = _summ(delta_iqc) if delta_iqc.size else {}

        instance = stats.get('instance', {}) or {}
        params = stats.get('parameters', {}) or {}

        # ---- 1) metadata json ---------------------------------------------
        metadata = {
            'run_id': run_id,
            'timestamp_utc': now.isoformat(),
            'location': location, 'key_location': key_location,
            'walking_profile': profile,
            'method_code': method_code, 'method_name': method_name, 'status': status,
            'budget': getattr(context, 'budget', None),
            'seeds': list(getattr(context, 'seeds', []) or []),
            'n_seeds': len(getattr(context, 'seeds', []) or []),
            'objective_metric': 'sum_iqc', 'optimization_direction': 'maximize',
            'baseline_objective': baseline_obj,
            'best_objective': best_obj,
            'improvement': improvement, 'improvement_pct': improvement_pct,
            'best_seed': summary.get('seed', stats.get('best_seed')),
            'runtime_seconds_total': stats.get('runtime_seconds_total'),
            'parameters': params,
            'instance': instance,
            'h3_resolution': h3_resolution, 'distance': distance,
            'input_files': input_files or {},
            'message': method_result.get('message'),
            'environment': {
                'python': sys.version.split()[0],
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'platform': platform.platform(),
                'git_commit': _git_commit(),
            },
        }
        meta_path = os.path.join(run_dir, 'run_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_default)
        paths['run_metadata'] = meta_path

        # ---- 2) one-row summary -------------------------------------------
        summary_row = {
            'run_id': run_id, 'timestamp_utc': now.isoformat(),
            'location': location, 'key_location': key_location, 'walking_profile': profile,
            'method_code': method_code, 'method_name': method_name, 'status': status,
            'budget': getattr(context, 'budget', None),
            'n_seeds': len(getattr(context, 'seeds', []) or []),
            'best_seed': summary.get('seed', stats.get('best_seed')),
            'objective_metric': 'sum_iqc', 'optimization_direction': 'maximize',
            'baseline_objective': baseline_obj, 'best_objective': best_obj,
            'improvement': improvement, 'improvement_pct': improvement_pct,
            'obj_seed_mean': seed_summ.get('mean'), 'obj_seed_std': seed_summ.get('std'),
            'obj_seed_min': seed_summ.get('min'), 'obj_seed_max': seed_summ.get('max'),
            'iqc_base_mean': base_summ.get('mean'), 'iqc_base_std': base_summ.get('std'),
            'iqc_opt_mean': opt_summ.get('mean'), 'iqc_opt_std': opt_summ.get('std'),
            'iqc_opt_min': opt_summ.get('min'), 'iqc_opt_median': opt_summ.get('median'),
            'iqc_opt_max': opt_summ.get('max'),
            'delta_iqc_mean': delta_summ.get('mean'), 'delta_iqc_std': delta_summ.get('std'),
            'delta_iqc_min': delta_summ.get('min'), 'delta_iqc_max': delta_summ.get('max'),
            'delta_iqc_positive_count': int(np.sum(delta_iqc > 0)) if delta_iqc.size else None,
            'n_hexagons': instance.get('n_hexagons'),
            'n_source_hexagons': instance.get('n_source_hexagons'),
            'n_candidate_dimensions': instance.get('n_candidate_dimensions'),
            'allocated_pois': summary.get('allocated_pois'),
            'distinct_pairs': summary.get('distinct_pairs'),
            'runtime_seconds_total': stats.get('runtime_seconds_total'),
            'rcl_alpha': params.get('rcl_alpha'),
            'construction_sample_size': params.get('construction_sample_size'),
            'local_search_max_evals': params.get('local_search_max_evals'),
            'improvement_eps': params.get('improvement_eps'),
            'parallel_workers': params.get('parallel_workers'),
            'eval_cache': params.get('eval_cache'),
            'h3_resolution': h3_resolution, 'distance': distance,
            'run_dir': run_dir,
        }
        df_summary = pd.DataFrame([summary_row])
        summary_path = os.path.join(run_dir, 'run_summary.csv')
        df_summary.reindex(columns=RUN_SUMMARY_COLUMNS).to_csv(summary_path, index=False, encoding='utf-8')
        _append_master(os.path.join(runs_root, 'all_runs_summary.csv'), df_summary, RUN_SUMMARY_COLUMNS)
        paths['run_summary'] = summary_path

        # human-readable report
        report_path = os.path.join(run_dir, 'run_report.txt')
        _write_report(report_path, summary_row, params, instance)
        paths['run_report'] = report_path

        ident = {
            'run_id': run_id, 'location': location, 'key_location': key_location,
            'walking_profile': profile, 'method_code': method_code,
        }

        # ---- 3) per-seed ---------------------------------------------------
        if per_seed:
            rows = []
            for s in per_seed:
                rows.append({**ident,
                             'seed': s.get('seed'),
                             'construction_objective': s.get('construction_objective'),
                             'local_search_objective': s.get('local_search_objective'),
                             'local_search_gain': s.get('local_search_gain'),
                             'improvement_over_baseline': s.get('improvement_over_baseline'),
                             'runtime_seconds': s.get('runtime_seconds'),
                             'distinct_pairs': s.get('distinct_pairs')})
            df_seed = pd.DataFrame(rows)
            seed_path = os.path.join(run_dir, 'per_seed.csv')
            df_seed.reindex(columns=PER_SEED_COLUMNS).to_csv(seed_path, index=False, encoding='utf-8')
            _append_master(os.path.join(runs_root, 'all_per_seed.csv'), df_seed, PER_SEED_COLUMNS)
            paths['per_seed'] = seed_path

            # convergence (construction trace per seed)
            conv_rows = []
            for s in per_seed:
                trace = s.get('trace') or []
                for step, obj in enumerate(trace, start=1):
                    conv_rows.append({'seed': s.get('seed'), 'step': step, 'objective': obj})
            if conv_rows:
                conv_path = os.path.join(run_dir, 'convergence.csv')
                pd.DataFrame(conv_rows).to_csv(conv_path, index=False, encoding='utf-8')
                paths['convergence'] = conv_path

        # ---- 4) per-hexagon IQC -------------------------------------------
        h3_ids = stats.get('h3_ids')
        if h3_ids is not None and iqc_base.size and iqc_opt.size:
            df_hex = pd.DataFrame({
                'h3_id': [str(h) for h in h3_ids],
                'iqc_baseline': iqc_base, 'iqc_optimized': iqc_opt,
                'delta_iqc': delta_iqc if delta_iqc.size else np.full(len(h3_ids), np.nan),
            })
            df_hex = _merge_hex_coords(df_hex, getattr(context, 'df_walkability', None))
            for k, v in ident.items():
                df_hex[k] = v
            hex_path = os.path.join(run_dir, 'per_hexagon_iqc.csv')
            df_hex.reindex(columns=PER_HEXAGON_COLUMNS).to_csv(hex_path, index=False, encoding='utf-8')
            _append_master(os.path.join(runs_root, 'all_per_hexagon_iqc.csv'), df_hex, PER_HEXAGON_COLUMNS)
            paths['per_hexagon_iqc'] = hex_path

        # ---- 5) per-hexagon indicators (tidy/long) ------------------------
        m_base = stats.get('indicator_matrix_baseline')
        m_opt = stats.get('indicator_matrix_optimized')
        ind_cols = stats.get('indicator_columns')
        if m_base is not None and m_opt is not None and ind_cols and h3_ids is not None:
            m_base = np.asarray(m_base, dtype=np.float64)
            m_opt = np.asarray(m_opt, dtype=np.float64)
            long_rows = []
            for r, h in enumerate(h3_ids):
                for c, ind in enumerate(ind_cols):
                    long_rows.append({
                        'h3_id': str(h), 'indicator': ind,
                        'value_baseline': float(m_base[r, c]),
                        'value_optimized': float(m_opt[r, c]),
                        'delta': float(m_opt[r, c] - m_base[r, c]),
                    })
            ind_path = os.path.join(run_dir, 'per_hexagon_indicators.csv')
            pd.DataFrame(long_rows).to_csv(ind_path, index=False, encoding='utf-8')
            paths['per_hexagon_indicators'] = ind_path

        # ---- 6) allocation -------------------------------------------------
        allocation = summary.get('allocation', []) or []
        if allocation:
            df_alloc = pd.DataFrame(allocation)
            for k, v in ident.items():
                df_alloc[k] = v
            df_alloc = _merge_hex_coords(df_alloc, getattr(context, 'df_walkability', None),
                                         keep_iqc=False)
            alloc_path = os.path.join(run_dir, 'allocation.csv')
            df_alloc.reindex(columns=ALLOCATION_COLUMNS).to_csv(alloc_path, index=False, encoding='utf-8')
            _append_master(os.path.join(runs_root, 'all_allocations.csv'), df_alloc, ALLOCATION_COLUMNS)
            paths['allocation'] = alloc_path

        # ---- 7) CRITIC weights --------------------------------------------
        w_base = stats.get('critic_weights_baseline')
        w_opt = stats.get('critic_weights_optimized')
        if w_base is not None and w_opt is not None and ind_cols:
            df_w = pd.DataFrame({
                'indicator': list(ind_cols),
                'weight_baseline': np.asarray(w_base, dtype=np.float64),
                'weight_optimized': np.asarray(w_opt, dtype=np.float64),
            })
            w_path = os.path.join(run_dir, 'critic_weights.csv')
            df_w.to_csv(w_path, index=False, encoding='utf-8')
            paths['critic_weights'] = w_path

        print(f"[results] Run persisted: {len(paths)} file(s) under {run_dir}")
    except Exception as exc:  # never break the optimization because of logging
        print(f"[results] Warning: could not persist run ({exc}).")

    return paths


def _merge_hex_coords(df: pd.DataFrame, df_walkability, keep_iqc: bool = True) -> pd.DataFrame:
    """Left-merge latitude/longitude (and pipeline IQC) by h3_id when available."""
    if df_walkability is None or 'h3_id' not in getattr(df_walkability, 'columns', []):
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        if keep_iqc:
            df['iqc_pipeline'] = np.nan
        return df
    cols = ['h3_id']
    for c in ('latitude', 'longitude'):
        if c in df_walkability.columns:
            cols.append(c)
    if keep_iqc and 'IQC' in df_walkability.columns:
        cols.append('IQC')
    ref = df_walkability[cols].copy()
    ref['h3_id'] = ref['h3_id'].astype(str)
    df['h3_id'] = df['h3_id'].astype(str)
    merged = df.merge(ref, on='h3_id', how='left')
    if keep_iqc:
        merged = merged.rename(columns={'IQC': 'iqc_pipeline'})
        if 'iqc_pipeline' not in merged.columns:
            merged['iqc_pipeline'] = np.nan
    for c in ('latitude', 'longitude'):
        if c not in merged.columns:
            merged[c] = np.nan
    return merged


def _write_report(path: str, row: dict, params: dict, instance: dict) -> None:
    def fmt(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v is not None else str(v)

    lines = [
        '=' * 70,
        'METAHEURISTIC OPTIMIZATION RUN REPORT',
        '=' * 70,
        f"Run ID            : {row.get('run_id')}",
        f"Timestamp (UTC)   : {row.get('timestamp_utc')}",
        f"Location          : {row.get('location')} (key: {row.get('key_location')})",
        f"Walking profile   : {row.get('walking_profile')}",
        f"Method            : {row.get('method_name')} ({row.get('method_code')})",
        f"Status            : {row.get('status')}",
        '-' * 70,
        f"Budget (POIs)     : {row.get('budget')}",
        f"Seeds (restarts)  : {row.get('n_seeds')}   best seed: {row.get('best_seed')}",
        f"Hexagons          : {instance.get('n_hexagons')} "
        f"(source: {instance.get('n_source_hexagons')})",
        f"Candidate dims    : {instance.get('n_candidate_dimensions')} "
        f"{instance.get('candidate_dimensions', '')}",
        '-' * 70,
        'OBJECTIVE (sum of IQC, maximize)',
        f"  baseline        : {fmt(row.get('baseline_objective'))}",
        f"  best            : {fmt(row.get('best_objective'))}",
        f"  improvement     : {fmt(row.get('improvement'))} "
        f"({fmt(row.get('improvement_pct'), 2)}%)",
        '-' * 70,
        'OBJECTIVE ACROSS SEEDS',
        f"  mean +/- std    : {fmt(row.get('obj_seed_mean'))} +/- {fmt(row.get('obj_seed_std'))}",
        f"  min / max       : {fmt(row.get('obj_seed_min'))} / {fmt(row.get('obj_seed_max'))}",
        '-' * 70,
        'IQC DISTRIBUTION (optimized, per hexagon)',
        f"  mean +/- std    : {fmt(row.get('iqc_opt_mean'))} +/- {fmt(row.get('iqc_opt_std'))}",
        f"  min/median/max  : {fmt(row.get('iqc_opt_min'))} / "
        f"{fmt(row.get('iqc_opt_median'))} / {fmt(row.get('iqc_opt_max'))}",
        f"  delta IQC mean  : {fmt(row.get('delta_iqc_mean'))} "
        f"(hexes improved: {row.get('delta_iqc_positive_count')})",
        '-' * 70,
        'PARAMETERS',
    ]
    for k, v in params.items():
        lines.append(f"  {k:<24}: {v}")
    lines += [
        '-' * 70,
        f"Allocated POIs    : {row.get('allocated_pois')} "
        f"in {row.get('distinct_pairs')} (hex, dim) pairs",
        f"Total runtime (s) : {fmt(row.get('runtime_seconds_total'), 3)}",
        '=' * 70,
        '',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
