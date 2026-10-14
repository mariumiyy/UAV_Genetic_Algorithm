import random
from chromosome import Chromosome
from fitness import evaluate_chromosome

class GeneticAlgorithm:
    def __init__(self, starts, blocked, hazards, priorities, population_size=50,
                 mutation_rate=0.10, generations=100, tournament_size=3,
                 elite_count=2, waypoint_count=3, seed=42):
        self.starts, self.blocked = starts, blocked
        self.hazards, self.priorities = hazards, priorities
        self.population_size, self.mutation_rate = population_size, mutation_rate
        self.generations, self.tournament_size = generations, tournament_size
        self.elite_count, self.waypoint_count = elite_count, waypoint_count
        random.seed(seed)
        self.best_history, self.average_history = [], []

    def initialize_population(self):
        return [Chromosome.random_chromosome(self.starts, self.blocked.shape[0],
                                              self.waypoint_count)
                for _ in range(self.population_size)]

    def evaluate(self, chromosome):
        f, m, r, valid = evaluate_chromosome(
            chromosome, self.blocked, self.hazards, self.priorities, self.starts)
        chromosome.fitness, chromosome.metrics = f, m
        chromosome.routes, chromosome.valid = r, valid
        return f

    def select_parent(self, population):
        candidates = random.sample(population, min(self.tournament_size, len(population)))
        return max(candidates, key=lambda x: x.fitness)

    def crossover(self, a, b):
        child = a.clone()
        for d in range(len(self.starts)):
            if random.random() < 0.5:
                child.waypoints[d] = b.waypoints[d][:]
                child.speeds[d] = b.speeds[d]
                child.sensors[d] = b.sensors[d][:]
            elif self.waypoint_count > 1:
                cut = random.randint(1, self.waypoint_count - 1)
                child.waypoints[d][cut:] = b.waypoints[d][cut:]
                child.sensors[d][cut:] = b.sensors[d][cut:]
        child.fitness = None
        return child

    def mutate(self, child):
        size = self.blocked.shape[0]
        for d in range(len(self.starts)):
            for j in range(self.waypoint_count):
                if random.random() < self.mutation_rate:
                    child.waypoints[d][j] = (random.randrange(1, size - 1),
                                              random.randrange(1, size - 1))
                if random.random() < self.mutation_rate:
                    child.sensors[d][j] = random.randint(0, 2)
            if random.random() < self.mutation_rate:
                child.speeds[d] = random.uniform(0.8, 1.4)
        child.fitness = None
        return child

    def run(self):
        population = self.initialize_population()
        best_overall = None
        for _ in range(self.generations):
            for c in population:
                if c.fitness is None:
                    self.evaluate(c)
            population.sort(key=lambda c: c.fitness, reverse=True)
            if best_overall is None or population[0].fitness > best_overall.fitness:
                best_overall = population[0].clone()
            scores = [c.fitness for c in population]
            self.best_history.append(population[0].fitness)
            self.average_history.append(sum(scores) / len(scores))
            new_population = [c.clone() for c in population[:self.elite_count]]
            while len(new_population) < self.population_size:
                child = self.crossover(self.select_parent(population),
                                       self.select_parent(population))
                self.mutate(child)
                self.evaluate(child)
                new_population.append(child)
            population = new_population
        for c in population:
            if c.fitness is None:
                self.evaluate(c)
        winner = max(population, key=lambda c: c.fitness)
        if winner.fitness > best_overall.fitness:
            best_overall = winner.clone()
        return best_overall
