import random
import time
import numpy as np

from environment import create_environment
from chromosome import Chromosome
from fitness import evaluate_chromosome
from genetic_algorithm import GeneticAlgorithm


def run_ga(pop_size, mutation_rate, tournament_size,
           generations=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    blocked, hazards, priorities, starts = create_environment()

    ga = GeneticAlgorithm(
        starts, blocked, hazards, priorities,
        population_size=pop_size,
        mutation_rate=mutation_rate,
        generations=generations,
        tournament_size=tournament_size,
        elite_count=2,
        waypoint_count=3,
        seed=seed,
    )
    t0 = time.time()
    best = ga.run()
    return best.fitness, time.time() - t0


def random_search(num_evaluations=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    blocked, hazards, priorities, starts = create_environment()
    grid_size = blocked.shape[0]

    best_fitness = -float("inf")
    t0 = time.time()
    for _ in range(num_evaluations):
        c = Chromosome.random_chromosome(
            starts, grid_size=grid_size, waypoint_count=3
        )
        f, m, r, valid = evaluate_chromosome(
            c, blocked, hazards, priorities, starts
        )
        c.fitness = f
        if f > best_fitness:
            best_fitness = f
    return best_fitness, time.time() - t0


def sweep(name, values, fixed, repeats=3):
    print(f"\n=== Sweeping {name} ===")
    print(f"{name:<18} | Best fitness (mean) | Std    | Runtime (s)")
    print("-" * 65)
    for v in values:
        fitnesses, runtimes = [], []
        for s in range(repeats):
            cfg = dict(fixed)
            cfg[name] = v
            cfg["seed"] = 42 + s
            f, t = run_ga(**cfg)
            fitnesses.append(f)
            runtimes.append(t)
        mean_f = float(np.mean(fitnesses))
        std_f = float(np.std(fitnesses))
        mean_t = float(np.mean(runtimes))
        print(f"{str(v):<18} | {mean_f:.4f}              | "
              f"{std_f:.4f} | {mean_t:.2f}")


if __name__ == "__main__":
    base = {"pop_size": 50, "mutation_rate": 0.10, "tournament_size": 3}

    sweep("pop_size", [20, 50, 100], base)
    sweep("mutation_rate", [0.01, 0.10, 0.30], base)
    sweep("tournament_size", [2, 3, 5], base)

    print("\n=== GA vs Random Search ===")
    fit_ga, time_ga = run_ga(pop_size=50, mutation_rate=0.10,
                             tournament_size=3)
    fit_rs, time_rs = random_search(num_evaluations=5000)

    print(f"GA best fitness:      {fit_ga:.4f}   (runtime {time_ga:.2f}s)")
    print(f"Random best fitness:  {fit_rs:.4f}   (runtime {time_rs:.2f}s)")
    print(f"GA improvement:       "
          f"{((fit_ga - fit_rs) / max(abs(fit_rs), 1e-6)) * 100:.1f}%")