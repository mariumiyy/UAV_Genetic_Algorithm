from collections import deque

def shortest_safe_segment(start, goal, blocked):
    rows, cols = blocked.shape
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols) or blocked[goal]:
        return None
    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or blocked[start]:
        return None
    queue = deque([start])
    parent = {start: None}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path, node = [], cell
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        for dr, dc in moves:
            nxt = (cell[0] + dr, cell[1] + dc)
            if (0 <= nxt[0] < rows and 0 <= nxt[1] < cols
                    and not blocked[nxt] and nxt not in parent):
                parent[nxt] = cell
                queue.append(nxt)
    return None

def build_route(start, waypoints, blocked):
    route, current = [start], start
    for waypoint in waypoints:
        segment = shortest_safe_segment(current, waypoint, blocked)
        if segment is None:
            return None
        route.extend(segment[1:])
        current = waypoint
    return route

def evaluate_chromosome(chromosome, blocked, hazards, priorities, starts,
                        energy_budget=120.0, time_budget=100.0, range_limit=100.0):
    routes, covered, detected = [], set(), set()
    total_distance = total_energy = total_time = total_risk = 0.0
    total_priority = float(priorities.sum())

    for i, start in enumerate(starts):
        route = build_route(start, chromosome.waypoints[i], blocked)
        if route is None:
            return -10.0, {"valid": False}, [], False
        routes.append(route)
        speed = max(0.5, min(2.0, chromosome.speeds[i]))
        for cell in route:
            covered.add(cell)
            row, col = cell
            total_risk += float(hazards[row, col])
            if priorities[row, col] > 0:
                for j, wp in enumerate(chromosome.waypoints[i]):
                    if wp == cell and chromosome.sensors[i][j] > 0:
                        detected.add(cell)
        distance = len(route) - 1
        total_distance += distance
        total_time += distance / speed
        total_energy += distance * (0.7 + 0.3 * speed)
        if distance > range_limit:
            return -10.0, {"valid": False, "reason": "range exceeded"}, [], False

    coverage = len(covered) / max(1, blocked.size - int(blocked.sum()))
    detection = sum(float(priorities[c]) for c in detected) / total_priority if total_priority else 0.0
    energy = min(1.0, total_energy / energy_budget)
    flight_time = min(1.0, total_time / time_budget)
    risk_denominator = max(1.0, float(hazards.max()) * sum(len(r) for r in routes))
    risk = min(1.0, total_risk / risk_denominator)
    fitness = 0.35 * coverage + 0.25 * detection - 0.15 * energy - 0.10 * flight_time - 0.15 * risk
    metrics = {"valid": True, "coverage": coverage, "detection": detection,
               "energy": total_energy, "flight_time": total_time,
               "risk": risk, "distance": total_distance}
    return fitness, metrics, routes, True
