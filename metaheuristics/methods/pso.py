import numpy as np
import numpy.typing as npt
import statistics

from ..core import build_final_indicator_matrix_nd, objective_function
from ..core.types import ObjectiveStateND, MetaheuristicContext
from dataclasses import dataclass
from math import floor
from numpy.random import default_rng, Generator
from pathlib import Path
from time import perf_counter

@dataclass
class Particle():
    """
    Represents a particle in the swarm, containing essential and auxiliary attributes.

    Attributes:
        x (set[int]): Selected POIs.
        objective (float): Value of the objective function.
        number (int): Number of selected POIs.
        pbest (set[int]): Best POIs found.
        pbest_obj (float): Value of the objective function for the best POIs.
    """
    
    x:         set[int]
    objective: float
    number:    int
    pbest:     set[int]
    pbest_obj: float

def generate_initial_swarm(
    size: int,
    budget: int,
    n_hex: int,
    n_pois: int,
    objective_state_nd: ObjectiveStateND,
    rng: Generator
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.uint8]]:
    """
    Populates the swarm with randomly generated particles. Each particle has a 50% chance of selecting an POI.

    Returns an array containing all the particles, and a 3-dimensional matrix. 
    The first dimension represents a particle, and the other two dimensions are the binary allocation matrix for that particle.
    
    # Note
    `pbest` and `pbest_obj` are left uninitialized (`set()` and `0.0` respectively). The caller is responsible for updating them,
    typically during the first pbest/gbest update step.

    Args:
        size (int): Swarm size.
        budget (int): Limitation on the number of allocations.
        n_hex (int): Number of hexagons.
        n_pois (int): Number of dimensions.
        objective_state_nd (ObjectiveStateND): Required metadata.
        rng (Generator): Random number generator.
    
    Returns:
        swarm (tuple[NDArray[object], NDArray[uint8]]): Particles and allocations.
    """
    
    swarm = np.empty(size, dtype = Particle)
    allocations = np.zeros((size, n_hex, n_pois), dtype = np.uint8)

    for i in range(size):
        particle = Particle(
            x          = {poi for poi in range(n_hex * n_pois) if rng.random() <= 0.5},
            objective  = 0.0,
            number     = 0,
            pbest      = set(),
            pbest_obj  = 0.0
        )

        for poi in particle.x:
            hs, ds = np.divmod(poi, n_pois)
            allocations[i][hs][ds] = 1
            particle.number += 1

        if particle.number > budget:
            particle.objective = 0
        else:
            matrix = build_final_indicator_matrix_nd(
                candidate_matrix = allocations[i],
                objective_state  = objective_state_nd
            )
            particle.objective = objective_function(final_indicator_matrix=matrix)["objective_value"]

        swarm[i] = particle

    return swarm, allocations

def scalar_multiplication(
    scalar: float,
    velocity: set[tuple[str, int]],
    rng: Generator
) -> set[tuple[str, int]]:
    """
    Multiplies a velocity by a scalar value.

    Selects `floor(scalar * |velocity|)` random elements from `velocity` and returns them.

    Args:
        scalar (float): Scaling factor in [0, 1].
        velocity (set[tuple[str, int]]): Set of (operator, POI) operations.
        rng (Generator): Random number generator.
    
    Returns:
        subset (set[tuple[str, int]]): Randomly sampled subset of the velocity.
    """

    # `scalar` is guaranteed to be in [0.0, 1.0] by the caller.
    if scalar < 0 or scalar > 1:
        return set()
    
    k = floor(scalar * len(velocity))
    
    samples = sorted(velocity)
    indexes = rng.choice(len(samples), size = k, replace = False)
    
    return {samples[i] for i in indexes}

def difference_in_positions(
    target: set[int],
    current: set[int]
) -> set[tuple[str, int]]:
    """
    Computes the velocity needed to transform `current` into `target`.

    Elements only in `target` become additions (`"+"`);
    elements only in `current` become removals (`"-"`).

    Args:
        target (set[int]): Desired position.
        current (set[int]): Position to be transformed.

    Returns:
        velocity (set[tuple[str, int]]): Set of (operator, POI) operations. 
    """
    
    additions = {("+", poi) for poi in target  - current}
    removals  = {("-", poi) for poi in current - target}
    return additions | removals

def number_of_elements(
    beta: float,
    set: set[int],
    rng: Generator
) -> int:
    """
    Returns the number of elements to operate on in a velocity update step.

    Uses stochastic rounding on `beta`: the result is `floor(beta)` with a probability of `fractional part of beta` of rounding 
    up to `ceil(beta)`.

    The result is clamped to `len(set)`.

    Args:
        beta (float): Expected number of elements.
        set (set[int]): Set whose size serves as the upper bound.
        rng (Generator): Random number generator.
    
    Returns:
        n_elements (int): Number of elements to operate on.
    """
    
    count = floor(beta)
    if rng.random() < beta - count:
        count += 1
    
    length = len(set)
    if count < length:
        return count
    return length

def k_tournament_selection(
    candidates: set[int],
    n_to_add: int,
    k: int,
    budget: int,
    number: int,
    allocation: npt.NDArray[np.uint8],
    objective_state_nd: ObjectiveStateND,
    rng: Generator
) -> set[tuple[str, int]]:
    """
    Greedily selects `n_to_add` POIs from `candidates` using tournament selection.

    For each slot, `k` candidates are sampled and the one that best improves the objective is chosen.

    Args:
        candidates (set[int]): POIs not present in `x union pbest union gbest`.
        n_to_add (int): Number of POIs to select.
        k (int): Tournament size.
        budget (int): Limitation on the number of allocations.
        number (int): Number of POIs for the current particle.
        allocation (NDArray[uint8]): Particle allocation matrix.
        objective_state_nd (ObjectiveStateND): Required metadata.
        rng (Generator): Random number generator.
        
    Returns:
        additions (set[tuple[str, int]]): Addition operations for the selected POIs.
    """
    
    additions = set()
    remaining = sorted(candidates)
    length    = len(remaining)

    running  = allocation.copy()
    r_number = number # Running number.
    _, n_pois = allocation.shape

    for _ in range(n_to_add):
        if k < length:
            tournament_size = k
        else:
            tournament_size = length
        
        indexes = rng.choice(length, size = tournament_size, replace = False)
        
        best_poi       = -1
        best_objective = -1
        
        for i in indexes:
            hs, ds = np.divmod(remaining[i], n_pois)

            if r_number + 1 > budget:
                objective = 0
            else:
                running[hs][ds] = 1
                matrix = build_final_indicator_matrix_nd(
                    candidate_matrix = running,
                    objective_state  = objective_state_nd
                )
                objective = objective_function(final_indicator_matrix=matrix)["objective_value"]
                running[hs][ds] = 0

            if objective > best_objective:
                best_poi       = remaining[i]
                best_objective = objective

        additions.add(("+", best_poi))
        remaining.remove(best_poi)
        length -= 1

        # Commit the selected POI for the next round.
        hs, ds = np.divmod(best_poi, n_pois)
        running[hs][ds] = 1
        r_number += 1

    return additions

def removal_of_elements(
    consensus: set[int],
    n_to_remove: int,
    rng: Generator
) -> set[tuple[str, int]]:
    """
    Randomly selects `n_to_remove` POIs from the `consensus` to remove.
    
    Args:
        consensus (set[int]): POIs present in `x intersect pbest intersect gbest`.
        n_ro_remove (int): Number of POIs to remove.
        rng (Generator): Random number generator.
    
    Returns:
        removals (set[tuple[str, int]]): Removal operations for the selected POIs.
    """

    samples = sorted(consensus)
    indexes = rng.choice(len(samples), size = n_to_remove, replace = False)

    return {("-", samples[i]) for i in indexes}

def sbpso(
    context: MetaheuristicContext,
    size: int,
    iterations: int,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    k: int,
    seed: int
) -> tuple[list[float], float, npt.NDArray[np.uint8]]:
    """
    Set-Based PSO for walkability problem. The algorithm performs binary operations; 
    that is, it either adds exactly one dimension of a given type to a hexagon, or it adds none.
    
    Furthermore, the set of POIs available for selection is considered to be the number of hexagons multiplied by the number of POIs. 
    The `divmod` operation is used to map a value from this set to a position in the intervention matrix.
    
    Args:
        context (MetaheuristicContext): Required metadata.
        size (int): Swarm size.
        iterations (int): Number of iterations.
        c1 (float): Cognitive acceleration coefficient (in [0, 1]).
        c2 (float): Social acceleration coefficient (in [0, 1]).
        c3 (float): Random addition coefficient.
        c4 (float): Random removal coefficient.
        k (int): Tournament size for k-tournament selection.
        seed (int): Randomness seed.
        
    Returns:
        data (tuple[list[float], float, NDArray[uint8]]): A tuple containing a list with the global best value for each iteration, 
        the total execution time, and the gbest's proposed matrix in the last iteration.
    """

    start = perf_counter()
    rng = default_rng(seed)

    # Parameters of the problem.
    n_hex    = len(context.source_hex_ids)
    n_pois    = len(context.dimensions)
    budget   = context.budget
    universe = set(range(n_hex * n_pois))

    # Initializing swarm.
    swarm, allocations = generate_initial_swarm(size, budget, n_hex, n_pois, context.objective_state_nd, rng)

    gbest     = set()
    gbest_obj = context.baseline_iqc_total

    objectives = []
    for _ in range(iterations):
        for i in range(size):
            particle = swarm[i]

            if particle.objective >= particle.pbest_obj:
                particle.pbest     = particle.x.copy()
                particle.pbest_obj = particle.objective

            if particle.objective >= gbest_obj:
                gbest     = particle.x.copy()
                gbest_obj = particle.objective

        objectives.append(gbest_obj)

        for i in range(size):
            particle = swarm[i]

            # Cognitive component: pull toward personal best.
            cognitive_velocity = scalar_multiplication(
                c1 * rng.random(),
                difference_in_positions(particle.pbest, particle.x),
                rng
            )

            # Social component: pull toward global best.
            social_velocity = scalar_multiplication(
                c2 * rng.random(),
                difference_in_positions(gbest, particle.x),
                rng
            )

            # Exploration: add POIs absent from all three reference sets.
            external_pois  = universe - (particle.x | particle.pbest | gbest)
            random_additions = k_tournament_selection(
                external_pois,
                number_of_elements(c3 * rng.random(), external_pois, rng),
                k,
                budget,
                particle.number,
                allocations[i],
                context.objective_state_nd,
                rng
            )

            # Diversity: remove POIs present in all three reference sets.
            consensus_pois = particle.x & particle.pbest & gbest
            random_removals  = removal_of_elements(
                consensus_pois,
                number_of_elements(c4 * rng.random(), consensus_pois, rng),
                rng
            )

            # Update the position.
            velocity = cognitive_velocity | social_velocity | random_additions | random_removals
            for op, poi in velocity:
                hs, ds = np.divmod(poi, n_pois)
                if op == "+":
                    particle.number += 1
                    particle.x.add(poi)
                    allocations[i][hs][ds] = 1
                else:
                    particle.number -= 1
                    particle.x.remove(poi)
                    allocations[i][hs][ds] = 0
            
            # Update objective value.
            if particle.number > budget:
                particle.objective = 0.0
            else:
                matrix = build_final_indicator_matrix_nd(
                    candidate_matrix = allocations[i],
                    objective_state  = context.objective_state_nd
                )
                particle.objective = objective_function(final_indicator_matrix=matrix)["objective_value"]
                
    end = perf_counter()

    # Building the gbest's proposed matrix.
    gbest_matrix = np.zeros((n_hex, n_pois), dtype = np.uint8)
    for e in gbest:
        hs, ds = np.divmod(e, n_pois)
        gbest_matrix[hs][ds] = 1

    return objectives, (end - start), gbest_matrix

def run_pso(
    context: MetaheuristicContext,
    mode: int = 1,
    size: int = 50,
    iterations: int = 600,
    c1: float = 0.9297,
    c2: float = 0.2266,
    c3: float = 1.3086,
    c4: float = 2.1526,
    k: int = 7
) -> dict:
    """
    Entry point for running the PSO. It serves as an interface for selecting the execution modes: single or benchmark.

    In single-run mode, the first random seed available in `context` is used, and the result of the method is returned. 
    In benchmark mode, all seeds are used, and the best result is returned, but all others are saved in the `results` directory.

    Args:
        context (MetaheuristicContext): Required metadata.
        mode (int): Execution mode (0: single; 1: benchmark) [default: 0].
        size (int): Swarm size [default: 50].
        iterations (int): Number of iterations [default: 600].
        c1 (float): Cognitive acceleration coefficient (in [0, 1]) [default: 0.9297].
        c2 (float): Social acceleration coefficient (in [0, 1]) [default: 0.2266].
        c3 (float): Random addition coefficient [default: 1.3086].
        c4 (float): Random removal coefficient [default: 2.1526].
        k (int): Tournament size for k-tournament selection [default: 7].
    """
    
    if not context.baseline_iqc_total:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "baseline_iqc_total is required for PSO execution."
        }

    if not context.objective_state_nd:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "objective_state_nd is required for PSO execution."
        }
    
    if not (0.0 <= c1 <= 1.0) or not (0.0 <= c2 <= 1.0):
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "c1 and c2 must be in the range from 0 to 1."
        }

    if mode == 0:
        objective, best_time, best_matrix = sbpso(context, size, iterations, c1, c2, c3, c4, k, context.seeds[0])
        best_objective = objective[len(objective) - 1]
        
        message = "Single execution completed successfully."
    else:
        # Hardcoded; ideally, it should be passed as a parameter.
        local = "av_paulista"
        result_path = Path("results") / "pso" / local

        result_path.mkdir(parents = True, exist_ok = True)
        if not (result_path / "runs.csv").is_file():
            with open(result_path / "runs.csv", "a") as file:
                file.write("walking profile, minimum value, maximum value, mean, standard deviation, mean time (s)\n")

        # The best value found across all runs.
        best_objective = 0.0
        best_matrix    = None
        best_time      = 0.0

        # All values found (for the statistics).
        objectives = []
        times      = []

        for s in context.seeds:
            objective, time, matrix = sbpso(context, size, iterations, c1, c2, c3, c4, k, s)
            obj = objective[len(objective) - 1]
            
            if obj >= best_objective:
                best_objective = obj
                best_matrix    = matrix        
                best_time      = time

            objectives.append(obj)
            times.append(time)
        
        # Saving the statistics in the following format: 
        # walking_profile, minimum value, maximum value, mean, standard deviation, and mean time.
        min_obj   = min(objectives)
        max_obj   = max(objectives)
        mean_obj  = statistics.mean(objectives)
        stdev_obj = statistics.stdev(objectives)
        mean_time = statistics.mean(times)

        with open(result_path / "runs.csv", "a+") as file:
            file.write(f"{context.walking_profile},{min_obj},{max_obj},{mean_obj},{stdev_obj},{mean_time}\n")

        # Saving all the results obtained.
        with open(result_path / "objectives.txt", "a+") as file:
            file.write(f"{context.walking_profile}: " + str(objectives) + "\n")
        
        message = "Benchmark completed successfully."

    return {
        "method_code": context.method_code,
        "method_name": context.method_name,
        "status": "ok",
        "objective_value": best_objective,
        "proposed_matrix": best_matrix,
        "total_time": best_time,
        "message": message,
    }