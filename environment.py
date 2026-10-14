import numpy as np

GRID_SIZE = 20

def create_environment():
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    hazards = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    priorities = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    blocked[8:12, 8:12] = True
    hazards[4:7, 13:16] = 0.8
    hazards[13:16, 9:12] = 0.5

    priorities[5, 5] = 1.0
    priorities[14, 15] = 0.8
    priorities[16, 5] = 0.6
    priorities[3, 15] = 0.7
    priorities[17, 17] = 0.9

    starts = [(1, 1), (1, 18), (18, 1)]
    return blocked, hazards, priorities, starts
