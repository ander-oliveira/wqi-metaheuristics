
# PCO212 Metaheuristics - Walkability + Metaheuristics Pipeline

  

## Overview

This project computes walkability indicators per H3 hexagon and calculates IQC (Walkability Quality Index). Mode `1` is used to generate the metaheuristic input datasets, and mode `2` is used to load existing inputs and run the metaheuristic stage.

  

Current goals:

- Select a location.

- Choose execution mode:

- run full pipeline for all walking profiles, or

- use an existing dataset for that location.

- Work with fixed distance `DISTANCE = 2000`.

- Save `df_walkability` and `hex_time_matrix` by default.

- Start metaheuristic setup only in mode `2`, after a dataset is loaded.

  


## Project Structure

```text
pco212-metaheuristics/
├── main.py                      # Main entry point
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── seeds.txt                    # Random seeds / experiment seeds
├── data/
│   ├── csv/
│   │   └── locations.csv        # Input location data
│   └── dem/
│       └── *.tif                # Digital elevation model files
├── walkability/                 # Walkability data processing pipeline
│   ├── __init__.py
│   ├── app.py
│   ├── common.py
│   ├── utils.py
│   ├── data_sources.py
│   ├── network_ops.py
│   ├── features.py
│   ├── hexagons.py
│   ├── indicators.py
│   ├── iqc.py
│   ├── meta_inputs.py
│   ├── persistence.py
│   ├── pipeline.py
│   └── visualization.py
└── metaheuristics/              # Optimization algorithms
    ├── __init__.py
    ├── optimizer.py
    ├── core/
    │   ├── __init__.py
    │   ├── io.py
    │   ├── budget.py
    │   ├── evaluation.py
    │   └── types.py
    └── methods/
        ├── __init__.py
        ├── ils.py
        ├── grasp.py
        ├── brkga.py
        ├── pso.py
        └── hybrid_grasp_vns_pr.py
```
  

## Module Responsibilities

### Entry points

-  `main.py`: minimal entry point, calls `walkability.app.run_cli()`.

-  `walkability/app.py`: main orchestrator for execution modes, profile loop, IQC pipeline, dataset generation (mode `1`), and metaheuristic stage trigger (mode `2`).

  

### Walkability modules

-  `walkability/common.py`: shared imports.

-  `walkability/utils.py`: location selection and base path utilities.

-  `walkability/data_sources.py`: DEM/POI/green-water/crosswalk/signal acquisition and cache.

-  `walkability/network_ops.py`: graph setup, Tobler time, node travel times.

-  `walkability/features.py`: real-time computation and accessibility filtering.

-  `walkability/hexagons.py`: H3 geometry and per-hexagon processing.

-  `walkability/indicators.py`: aggregated indicators (`S_*`, `A_*`, `I_*`, `T_*`, `U_*`).

-  `walkability/iqc.py`: CRITIC weights and final IQC computation.

-  `walkability/meta_inputs.py`: generation of source-target temporal impact matrix (`hex_time_matrix`) using Tobler-aware travel time and cosine decay (`alpha_20`).

-  `walkability/persistence.py`: CSV persistence with output flags.

-  `walkability/pipeline.py`: step-based pipeline and per-hexagon worker logic.

-  `walkability/visualization.py`: all map/heatmap plotting logic.

  

### Metaheuristics modules

-  `metaheuristics/optimizer.py`:

- method selection prompt (A-E),

- orchestrates seed loading, random allocation setup, and method dispatch,

- public API: `walk_meta_opt(df_walkability, df_hex_time_matrix, budget, method, seeds, walking_profile)`.

-  `metaheuristics/core/io.py`: seed-file loading utilities.

-  `metaheuristics/core/budget.py`: budget allocation helpers and POI dimension constants (including spatial allocations by `(hex, dimension)`).

-  `metaheuristics/core/evaluation.py`: shared evaluation helpers (baseline IQC stats, spatial-time allocation application, CRITIC/IQC recomputation).

- includes the common objective functions for all methods:

-  `recalculate_iqc_and_critic(df_final)`

-  `objective_function(final_indicator_matrix=...)` (official objective entry point: CRITIC + IQC + sum(IQC))

-  `build_final_indicator_matrix_nd(candidate_matrix, objective_state)` (builds baseline+proposal matrix with source-target `alpha_20` impact)

-  `build_objective_state_nd(df_walkability, df_hex_time_matrix, candidate_dimensions)` (one-time precompilation to ndarray state)

-  `evaluate_candidate_matrix_nd(candidate_matrix, objective_state)` (compatibility wrapper: candidate -> final matrix -> objective)

-  `metaheuristics/core/types.py`: shared dataclass context for method implementations.

-  `metaheuristics/methods/*.py`: one module per metaheuristic method implementation.

-  `metaheuristics/methods/__init__.py`: method registry (`METHOD_RUNNERS`) used by the optimizer.

-  `metaheuristics/methods/README.md`: quick start for method implementers.

  

## Runtime Flow (app.py)

1. Prompt execution mode:

-  `1`: run full pipeline for all walking profiles.

-  `2`: use existing dataset for selected location.

2. Select one location or all locations from `data/csv/locations.csv`.

3. If mode `1`:

- run profiles (`average_adult`, `elderly`, `athlete`) with data reuse,

- compute IQC,

- save `df_walkability` datasets per profile/location,

- save `hex_time_matrix` datasets per profile/location (source-target hex temporal impact),

- finish without opening metaheuristic prompts.

4. If mode `2`:

- search existing `df_walkability` CSVs for that location/profile,

- list available files,

- load selected `df_walkability` and matching `hex_time_matrix` into memory,

- start metaheuristic setup stage:

- choose method,

- load seeds list from `seeds.txt`,

- use `BUDGET = 100`,

- call `walk_meta_opt(...)`.

## Metaheuristic Inputs (Why Two Files)
The metaheuristic stage requires **two complementary inputs** for each `(location, profile)`:

1. `df_walkability` (`*_walkability_index_*.csv`)
- Contains one row per target hexagon with:
- identifiers (`h3_id`, `latitude`, `longitude`),
- 11 indicators (`S_*`, `I_*`, `A_*`, `C_*`, `T_*`, `U_*`),
- baseline `IQC`.

2. `df_hex_time_matrix` (`*_hex_time_matrix_*.csv`)
- Contains source-target temporal impact pairs:
- `source_h3_id`, `target_h3_id`, `time_min`, `alpha_20`, plus node/profile metadata.

Why both are required:
- `df_walkability` is the optimization **state** (what values will be changed and reevaluated).
- `df_hex_time_matrix` is the spatial-time **impact model** (how POI insertion in one source hex affects many target hexagons through Tobler-aware travel times).
- Without the second file, the model cannot apply distance/time decay impacts consistently across hexagons.

  

## Metaheuristics Methods (current options)

- A) Iterated Local Search (ILS)

- B) Greedy Randomized Adaptive Search Procedure (GRASP)

- C) Biased Random-Key Genetic Algorithm (BRKGA)

- D) Particle Swarm Optimization (PSO)

- E) Hybrid Methods (GRASP + VNS + Path Relinking)

  

## Current Output Policy

Configured in `walkability/app.py`:

-  `SAVE_ONLY_DF_WALKABILITY = True`

-  `SAVE_CRITIC_WEIGHTS_CSV = False`

-  `SAVE_INDICATORS_BASE_CSV = False`

-  `SAVE_HEX_TIME_MATRIX_CSV = True`

  

Practical result:

- `df_walkability` and `hex_time_matrix` CSVs are persisted (core inputs for mode `2`).

  

## Optimization Objective

- Global objective for metaheuristics: **maximize total IQC** after applying spatial-time allocation impacts.

- Implemented as `sum_iqc` (sum of IQC across all hexagons).

- Fast path: use `build_final_indicator_matrix_nd(...)` + `objective_function(final_indicator_matrix=...)` with a precompiled `objective_state_nd`.


  

## Run

```bash

pip  install  -r  requirements.txt

python  main.py

```

  

## Instructions to Implement a Metaheuristic Method



### 1) Method contract (required)

Each method module in `metaheuristics/methods/` must expose one function:

  

```python

def run_<method_name>(context: MetaheuristicContext) -> dict:

...

```

  

The `context` object (from `metaheuristics/core/types.py`) provides:

-  `df_walkability`: current dataframe in memory.

-  `df_hex_time_matrix`: source-target temporal impact matrix.

-  `budget`: number of POIs to allocate.

-  `method_code`, `method_name`: selected method metadata.

-  `seeds`: list of seeds loaded from `seeds.txt`.

-  `walking_profile`: selected profile key.

-  `dimensions`: POI dimensions available in the dataframe.

-  `source_hex_ids`: source hexagons eligible to receive inserted POIs.

-  `baseline_iqc_total`: baseline global IQC (sum) before optimization.

-  `allocations`: initial random allocations generated from seeds.

-  `objective_state_nd`: precompiled numeric objective state (`h3_id -> index`, `baseline_matrix`, source-target alpha arrays).

What `objective_state_nd` is:
- It is the one-time compiled representation of optimization inputs for fast objective evaluation.
- It stores: sequential hex indexing (`h3_id -> idx`), `baseline_matrix`, dimension indexing, and source-target impact arrays (`source_idx`, `target_idx`, `alpha`).
- It is built once by `build_objective_state_nd(...)` before the metaheuristic loop.
- The method hot loop builds `final_indicator_matrix` from `candidate_matrix` + `objective_state_nd` and then evaluates it with `objective_function(...)`, avoiding dataframe `merge/groupby/pivot` overhead.

Plain-language note (no OOP background required):

- `MetaheuristicContext` is just a "data package" created before your method starts.

- Think of it as a ready-to-use input object: your method receives `context` and reads fields like `context.df_walkability`, `context.df_hex_time_matrix`, and `context.allocations`.

- You do **not** need to instantiate this class inside `run_ils`/`run_grasp`/etc.; `walk_meta_opt(...)` creates it and passes it automatically.

- In practice, `context.allocations[0]` means "first initial candidate allocation generated from seeds", used as a starting point.

  

### 2) Objective function (common to all methods)

All methods must evaluate candidate solutions with:

  

```python

from metaheuristics.core import (
    allocation_items_to_candidate_matrix,
    build_final_indicator_matrix_nd,
    objective_function,
)

  

candidate_matrix = allocation_items_to_candidate_matrix(
    allocation_items=allocation_items,
    objective_state=context.objective_state_nd,
)

final_indicator_matrix = build_final_indicator_matrix_nd(
    candidate_matrix=candidate_matrix,
    objective_state=context.objective_state_nd,
)

result = objective_function(
    final_indicator_matrix=final_indicator_matrix,
)

```

  

`objective_function(final_indicator_matrix=...)`:

- expects a ready final indicator matrix,

- recalculates CRITIC weights,

- recalculates IQC for every hexagon,

- returns `objective_value = sum(IQC)` and `optimization_direction = "maximize"`.

  

This is the global objective: **maximize total IQC**.

  

### 3) What a method should return

Return a dictionary with at least:

-  `method_code`

-  `method_name`

-  `status` (`ok`, `placeholder`, `error`, etc.)

-  `best_objective_value` (when implemented)

-  `best_solution_summary` (optional but recommended)

-  `message` (human-readable status)

Current temporary debug fields (ndarray control exports):
- `debug_baseline_matrix_file`
- `debug_initial_candidate_matrix_file`

  

Example shape:

```python

{

"method_code": "A",

"method_name": "Iterated Local Search (ILS)",

"status": "ok",

"best_objective_value": 123.456,

"best_solution_summary": {"seed": 182, "iterations": 200},

"message": "ILS finished successfully."

}

```

  

### 4) Registering a new method

1. Implement the method in its module (`metaheuristics/methods/<name>.py`).

2. Export/import it in `metaheuristics/methods/__init__.py`.

3. Register it in `METHOD_RUNNERS` with its option code (`A` to `E`).

  

Without registration, the optimizer cannot dispatch to the method.

  

### 5) Minimal implementation example

```python

from ..core.types import MetaheuristicContext

from ..core import (
    allocation_items_to_candidate_matrix,
    build_final_indicator_matrix_nd,
    objective_function,
)

  

def run_ils(context: MetaheuristicContext) -> dict:

first_candidate = context.allocations[0]

candidate_matrix = allocation_items_to_candidate_matrix(
    allocation_items=first_candidate["allocation"],
    objective_state=context.objective_state_nd,
)

final_indicator_matrix = build_final_indicator_matrix_nd(
    candidate_matrix=candidate_matrix,
    objective_state=context.objective_state_nd,
)

eval_result = objective_function(
    final_indicator_matrix=final_indicator_matrix,
)

return {

"method_code": context.method_code,

"method_name": context.method_name,

"status": "ok",

"best_objective_value": eval_result["objective_value"],

"message": "Baseline spatial-time evaluation executed."

}

```

## Detailed Git Contribution Workflow (Implementing a New Metaheuristic Method)

This section describes the recommended collaboration flow when someone needs to implement or improve a method (`ILS`, `GRASP`, `BRKGA`, etc.) in this repository.

### Why this workflow

- Prevents unstable code from going directly into `main`.
- Keeps one branch focused on one method/feature.
- Makes code review and reproducibility easier.
- Preserves a clear project history.

### Choose your contribution path first

Use one of the two paths below.

1. You have write access to this repository:
- work directly from this repository using a feature branch.

2. You do not have write access:
- create a fork, work in your fork, and open a Pull Request to this repository.

Both paths end in a Pull Request (PR) for review before merge.

### One-time local Git setup (if needed)

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Step-by-step (with write access)

1. Clone the repository

```bash
git clone <REPO_URL>
cd pco212-metaheuristics
```

2. Ensure `main` is up to date

```bash
git checkout main
git pull origin main
```

3. Create a branch for your method

Recommended naming:
- `feat/metaheuristic-<method-name>`
- `fix/metaheuristic-<method-name>`
- `exp/metaheuristic-<method-name>` (for experiments)

Examples:
- `feat/metaheuristic-ils`
- `feat/metaheuristic-grasp`
- `feat/metaheuristic-brkga`

Command:

```bash
git checkout -b feat/metaheuristic-brkga
```

4. Implement the method

Typical files touched:
- `metaheuristics/methods/<method>.py`
- `metaheuristics/methods/__init__.py` (method registration)
- optional documentation updates in `README.md` and/or `metaheuristics/methods/README.md`

5. Validate locally before commit

At minimum:
- run your local checks/tests.
- run at least one smoke execution to ensure the method is callable from the optimizer path.

Example baseline checks:

```bash
python -m compileall metaheuristics walkability
python main.py
```

6. Commit in small, meaningful units

Check what changed:

```bash
git status
git diff
```

Stage and commit:

```bash
git add metaheuristics/methods/brkga.py metaheuristics/methods/__init__.py README.md
git commit -m "feat: implement BRKGA metaheuristic runner"
```

Good commit message prefixes:
- `feat:` new functionality
- `fix:` bug fix
- `refactor:` internal restructuring without behavior change
- `docs:` documentation only
- `test:` tests only

7. Push your branch

```bash
git push -u origin feat/metaheuristic-brkga
```

8. Open a Pull Request

Open PR:
- base branch: `main`
- compare branch: your feature branch

PR should include:
- objective of the method/changes,
- files changed,
- how to run and validate,
- known limitations,
- sample output/metrics when relevant.

9. Request review and address feedback

- Ask maintainers/reviewers explicitly.
- Push new commits to the same branch; the PR updates automatically.
- Keep discussion and technical decisions in PR comments for traceability.

10. Merge after approval

Preferred merge policy:
- `Squash and merge` for a cleaner `main` history.

After merge:

```bash
git checkout main
git pull origin main
git branch -d feat/metaheuristic-brkga
git push origin --delete feat/metaheuristic-brkga
```

### Step-by-step (without write access: fork flow)

1. Fork this repository in GitHub/GitLab UI.
2. Clone your fork:

```bash
git clone <YOUR_FORK_URL>
cd pco212-metaheuristics
```

3. Add original repository as `upstream`:

```bash
git remote add upstream <ORIGINAL_REPO_URL>
git fetch upstream
git checkout main
git pull upstream main
```

4. Create your branch from updated `main`:

```bash
git checkout -b feat/metaheuristic-grasp
```

5. Implement, validate, commit, and push to your fork:

```bash
git push -u origin feat/metaheuristic-grasp
```

6. Open PR from:
- `your-fork/feat/metaheuristic-grasp` -> `original-repo/main`

### Collaboration rules recommended for this project

- Never commit directly to `main`.
- One method/feature per branch.
- Keep PRs focused and reasonably small.
- Update docs when behavior changes.
- Keep your branch updated if `main` moves while your PR is open:

```bash
git checkout main
git pull origin main
git checkout feat/metaheuristic-ils
git merge main
```

If a PR gets large, split into multiple PRs:
1. `PR 1`: refactor/preparation
2. `PR 2`: method logic
3. `PR 3`: tuning/metrics/docs

### Fast command checklist

```bash
# Start work
git checkout main
git pull origin main
git checkout -b feat/metaheuristic-ils

# During work
git status
git add .
git commit -m "feat: add initial ILS neighborhood search"

# Publish
git push -u origin feat/metaheuristic-ils

# After PR merge
git checkout main
git pull origin main
git branch -d feat/metaheuristic-ils
```

### Direct answer to the common question

Yes: the expected flow is:
- clone (or fork + clone),
- create a branch (preferably named after the method),
- implement and commit in that branch,
- open a Pull Request,
- request review,
- merge only after approval.
