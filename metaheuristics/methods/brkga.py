import numpy as np
import numpy.typing as npt
import statistics

from ..core import build_final_indicator_matrix_nd, objective_function
from ..core.types import MetaheuristicContext, ObjectiveStateND
from dataclasses import dataclass
from math import floor
from numpy.random import default_rng, Generator
from pathlib import Path
from time import perf_counter
from typing import cast

@dataclass(repr = False, eq = False, match_args = False)
class Solution:
    """
    Contains the attributes of a BRKGA solution.
    Useful for sorting, since the genes and the objective value are grouped together.

    Attributes:
        genes (NDArray[float64]): Vector of random keys.
        objective (float): Value of the objective function.
    """
    
    genes:     npt.NDArray[np.float64]
    objective: float

    def decoder(self, n_hex: int, n_pois: int, k: int, objective_state_nd: ObjectiveStateND):
        """
        Sort the vector of random keys and select the top `k` (their indexes before sorting).
        Using the new indexes, build the allocation matrix and calculate the new fitness value.

        The random key vector is expected to have a size of `n_hex * n_pois`.

        Args:
            n_hex (int): Number of hexagons.
            n_pois (int): Number of dimensions.
            k (int): Number of allocations.
            objective_state_nd (ObjectiveStateND): Required metadata.
        """

        allocations = np.zeros((n_hex, n_pois), dtype = np.uint8)
        top_k       = np.argsort(self.genes)[-k:]
        
        for poi in top_k:
            hs, ds = np.divmod(poi, n_pois)
            allocations[hs][ds] = 1
        
        matrix = build_final_indicator_matrix_nd(
            candidate_matrix = allocations,
            objective_state  = objective_state_nd
        )
        self.objective = objective_function(final_indicator_matrix=matrix)["objective_value"]

def generate_initial_population(
    size: int,
    budget: int,
    n_hex: int,
    n_pois: int,
    objective_state_nd: ObjectiveStateND,
    rng: Generator
) -> list[Solution]:
    """
    Initializes the population with random key vectors.

    Args:
        size (int): Population size.
        budget (int): Number of allocations.
        n_hex (int): Number of hexagons.
        n_pois (int): Number of dimensions.
        objective_state_nd (ObjectiveStateND): Required metadata.
        rng (Generator): Random number generator.

    Returns:
        population (list[Solution]): Solutions.
    """

    population = []

    universe = n_hex * n_pois
    for _ in range(size):
        solution = Solution(
            genes     = rng.random(universe),
            objective = 0.0
        )

        solution.decoder(n_hex, n_pois, budget, objective_state_nd)
        population.append(solution)

    population.sort(key = lambda x: x.objective, reverse = True)
    return population

def brkga(
    context: MetaheuristicContext,
    size: int,
    iterations: int,
    elite_size: float,
    mutant_size: float,
    inheritance: float,
    seed: int
) -> tuple[list[float], float, npt.NDArray[np.uint8]]:
    """
    BRKGA for walkability problem. There are no parallel populations or restart mechanisms. 

    The decoder sorts the vector of random keys and selects the top `k`, where `k` is the budget.
    To do this, the algorithm maps all possible allocations to all hexagons in a range from 0 to `n_hex * n_pois - 1`.

    Thus, the size of the random key vector is `n_hex * n_pois`.

    Since the allocations are expected to be in a binary matrix of shape `(n_hex X n_pois)`,
    the dimension `d` is mapped to rows and columns in the matrix as follows:
    - Row: `floor(d / n_pois)`
    - Column: `d % n_pois`

    Args:
        context (MetaheuristicContext): Required metadata.
        size (int): Population size.
        iterations (int): Number of iterations.
        elite_size (float): Size of the elite partition.
        mutant_size (float): Size of the mutant partition.
        inheritance (float): Child inheritance probability.
        seed (int): Randomness seed.

    Returns:
        data (tuple[list[float], float, NDArray[uint8]]): A tuple containing a list with the best value for each iteration, 
        the total execution time, and the best proposed matrix in the last iteration.
    """

    start = perf_counter()
    rng = default_rng(seed)
    context.objective_state_nd = cast(ObjectiveStateND, context.objective_state_nd)

    # Parameters of the problem.
    n_hex    = len(context.source_hex_ids)
    n_pois   = len(context.dimensions)
    universe = n_hex * n_pois
    budget   = context.budget

    # Population segments (elite | normal | mutants). Indicate the final index (+1) for each segment.
    elite = floor(size * elite_size)
    normal = size - floor(size * mutant_size)
    mutant = size

    # Initializing population.
    population = generate_initial_population(size, budget, n_hex, n_pois, context.objective_state_nd, rng)
    next_population = [
        Solution(genes=np.zeros(universe, dtype=np.float64), objective=0.0)
        for _ in range(size)
    ]

    objectives = [0.0] * iterations
    for i in range(iterations):
        objectives[i] = population[0].objective

        # Copy elite solutions to the next generation.
        for e in range(elite):
            next_population[e].genes[:]  = population[e].genes
            next_population[e].objective = population[e].objective

        # Randomly selects one parent from the elite partition and one parent from the non-elite population
        # and generates a new solution.
        for n in range(elite, normal):
            parent1 = rng.integers(0, elite)
            parent2 = rng.integers(elite, normal)

            mask = rng.random(universe) <= inheritance
            next_population[n].genes[:] = np.where(mask, population[parent1].genes, population[parent2].genes)
            next_population[n].decoder(n_hex, n_pois, budget, context.objective_state_nd)

        # Generates new random solutions.
        for m in range(normal, mutant):
            next_population[m].genes[:] = rng.random(universe)
            next_population[m].decoder(n_hex, n_pois, budget, context.objective_state_nd)
            
        population, next_population = next_population, population
        population.sort(key = lambda x: x.objective, reverse = True)

    end = perf_counter()

    # Building final matrix.
    allocations = np.zeros((n_hex, n_pois), dtype = np.uint8)
    top_k       = np.argsort(population[0].genes)[-budget:]

    for poi in top_k:
        hs, ds = np.divmod(poi, n_pois)
        allocations[hs][ds] = 1

    return objectives, (end - start), allocations

def run_brkga(
    context: MetaheuristicContext,
    mode: int = 0,
    size: int = 100,
    iterations: int = 600,
    elite_size: float = 0.15,
    mutant_size: float = 0.10,
    inheritance: float = 0.50
) -> dict:
    """
    Entry point for running the BRKGA. It serves as an interface for selecting the execution modes: single or benchmark.

    In single-run mode, the first random seed available in `context` is used, and the result of the method is returned. 
    In benchmark mode, all seeds are used, and the best result is returned, but all others are saved in the `results` directory.

    Args:
        context (MetaheuristicContext): Required metadata.
        mode (int): Execution mode (0: single; 1: benchmark) [default: 0].
        size (int): Population size [default: 100].
        iterations (int): Number of iterations [default: 600].
        elite_size (float): Size of the elite partition (10% - 25%) [default: 0.15].
        mutant_size (float): Size of the mutant partition (10% - 30%) [default: 0.10].
        inheritance (float): Child inheritance probability (50% - 80%) [default: 0.50].
    """

    if not context.objective_state_nd:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "objective_state_nd is required for BRKGA execution."
        }
    
    if elite_size < 0.0 or mutant_size < 0.0:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "The size of partitions cannot be negative."
        }

    if elite_size + mutant_size > 1.0:
        return {
            "method_code": context.method_code,
            "method_name": context.method_name,
            "status": "error",
            "message": "The sum of the partition sizes cannot exceed 1.0."
        }

    if mode == 0:
        objective, best_time, best_matrix = brkga(context, size, iterations, elite_size, mutant_size, inheritance, context.seeds[0])
        best_objective = objective[len(objective) - 1]

        message = "Single execution completed successfully."
    else:
        # Hardcoded; ideally, it should be passed as a parameter.
        local = "av_paulista"
        result_path = Path("results") / "brkga" / local

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
            objective, time, matrix = brkga(context, size, iterations, elite_size, mutant_size, inheritance, s)
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