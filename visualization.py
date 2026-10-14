import os
import numpy as np
import matplotlib.pyplot as plt


def _as_list(collection):
    """Safely turn numpy arrays / None / lists into a Python list."""
    if collection is None:
        return []
    if isinstance(collection, np.ndarray):
        return collection.tolist()
    if isinstance(collection, (list, tuple, set)):
        return list(collection)
    return [collection]


def _grid_size_from_data(blocked, hazards, priorities, starts):
    coords = []

    for coll in (_as_list(blocked), _as_list(hazards), _as_list(starts)):
        for item in coll:
            if isinstance(item, (tuple, list, np.ndarray)) and len(item) >= 2:
                coords.append(int(item[0]))
                coords.append(int(item[1]))

    if isinstance(priorities, dict):
        keys = list(priorities.keys())
    else:
        keys = _as_list(priorities)

    for item in keys:
        if isinstance(item, (tuple, list, np.ndarray)) and len(item) >= 2:
            coords.append(int(item[0]))
            coords.append(int(item[1]))

    if coords:
        return max(20, max(coords) + 2)
    return 20


def save_visualizations(best_history, average_history, best,
                        blocked, hazards, priorities, starts,
                        output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    grid = _grid_size_from_data(blocked, hazards, priorities, starts)

    # ------------------------------------------------------------
    # 1. Fitness convergence
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(list(best_history), label="Best fitness", linewidth=2)
    plt.plot(list(average_history), label="Average fitness",
             linewidth=1.5, alpha=0.8)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Fitness Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitness_convergence.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # 2. Drone routes map
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 8))

    # Blocked cells
    for b in _as_list(blocked):
        if isinstance(b, (tuple, list, np.ndarray)) and len(b) >= 2:
            plt.scatter(b[0], b[1], c="black", marker="s", s=90)

    # Hazards
    for h in _as_list(hazards):
        if isinstance(h, (tuple, list, np.ndarray)) and len(h) >= 2:
            plt.scatter(h[0], h[1], c="red", marker="x", s=90)

    # Targets
    if isinstance(priorities, dict):
        target_list = list(priorities.keys())
    else:
        target_list = _as_list(priorities)
    for t in target_list:
        if isinstance(t, (tuple, list, np.ndarray)) and len(t) >= 2:
            plt.scatter(t[0], t[1], c="green", marker="*", s=180)

    # Starting positions
    for s in _as_list(starts):
        if isinstance(s, (tuple, list, np.ndarray)) and len(s) >= 2:
            plt.scatter(s[0], s[1], c="blue", marker="o", s=90)

    # Routes — use best.routes if available
    routes = getattr(best, "routes", None) or []
    colors = ["tab:blue", "tab:orange", "tab:purple",
              "tab:cyan", "tab:magenta", "tab:brown"]

    for i, route in enumerate(routes):
        if not route:
            continue
        xs = [int(p[0]) for p in route]
        ys = [int(p[1]) for p in route]
        plt.plot(xs, ys, marker=".", markersize=4,
                 color=colors[i % len(colors)],
                 label=f"Drone {i + 1}")

    plt.xlim(-0.5, grid - 0.5)
    plt.ylim(-0.5, grid - 0.5)
    plt.grid(True, alpha=0.3)
    plt.title("UAV Routes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "uav_routes.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # 3. Coverage heatmap
    # ------------------------------------------------------------
    coverage = np.zeros((grid, grid))
    for route in routes:
        for p in route:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < grid and 0 <= y < grid:
                coverage[y, x] = 1

    plt.figure(figsize=(8, 7))
    plt.imshow(coverage, origin="lower", cmap="YlGn",
               extent=(-0.5, grid - 0.5, -0.5, grid - 0.5))
    plt.colorbar(label="Covered (1 = yes)")
    plt.title("Coverage Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coverage_heatmap.png"), dpi=150)
    plt.close()

    print(f"Visualizations saved to {output_dir}/")