import random
import numpy as np
from environment import create_environment
from genetic_algorithm import GeneticAlgorithm
from visualization import save_visualizations

def main():
    random.seed(42)
    np.random.seed(42)
    blocked, hazards, priorities, starts = create_environment()
    ga = GeneticAlgorithm(starts, blocked, hazards, priorities,
                          population_size=50, mutation_rate=0.10,
                          generations=100, tournament_size=3,
                          elite_count=2, waypoint_count=3, seed=42)
    best = ga.run()
    print("\n=== BEST UAV DEPLOYMENT PLAN ===")
    print(f"Best fitness: {best.fitness:.4f}")
    print(f"Valid route: {best.valid}")
    print("Metrics:")
    for name, value in best.metrics.items():
        print(f"  {name}: {value}")
    for i, route in enumerate(best.routes):
        print(f"\nDrone {i+1}")
        print(f"  Waypoints: {best.waypoints[i]}")
        print(f"  Speed multiplier: {best.speeds[i]:.2f}")
        print(f"  Sensor settings: {best.sensors[i]}")
        print(f"  Route cells: {len(route)}")
    save_visualizations(ga.best_history, ga.average_history, best,
                        blocked, hazards, priorities, starts)
    print("\nSaved graphs in outputs/")

if __name__ == "__main__":
    main()
