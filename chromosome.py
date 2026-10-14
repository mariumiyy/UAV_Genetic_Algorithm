import copy
import random

class Chromosome:
    def __init__(self, waypoints, speeds, sensors):
        self.waypoints = waypoints
        self.speeds = speeds
        self.sensors = sensors
        self.fitness = None
        self.metrics = {}
        self.routes = []
        self.valid = False

    def clone(self):
        return copy.deepcopy(self)

    @classmethod
    def random_chromosome(cls, starts, grid_size=20, waypoint_count=3):
        waypoints, speeds, sensors = [], [], []
        for _ in starts:
            waypoints.append([
                (random.randrange(1, grid_size - 1),
                 random.randrange(1, grid_size - 1))
                for _ in range(waypoint_count)
            ])
            speeds.append(random.uniform(0.8, 1.4))
            sensors.append([random.randint(0, 2) for _ in range(waypoint_count)])
        return cls(waypoints, speeds, sensors)
