
import json
import random
import asyncio
import heapq
import numpy as np
import time
from typing import List, Dict, Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# QUBO Imports
import openjij as oj
import dimod

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Structures ---

class Point(BaseModel):
    id: int
    x: float
    y: float

class Agent(BaseModel):
    id: str
    idx: int
    color: str
    is_cpu: bool = False

class GameState:
    def __init__(self):
        self.points: List[Point] = []
        self.edges: Dict[int, List[int]] = {}
        self.destinations: List[int] = []
        self.visited_destinations: Set[int] = set()
        self.agents: Dict[str, Agent] = {}
        self.depot_index: int = 0
        self.suggested_routes: Dict[str, List[int]] = {} 
        self.status: str = "LOBBY"
        
        self.num_points = 80
        self.num_destinations = 8
        self.k_neighbors = 4
        
        self.generate_map()

    def generate_map(self):
        width, height = 800, 600
        margin = 50
        self.points = []
        min_dist = 35
        
        for i in range(self.num_points):
            valid_point = False
            for _ in range(20):
                x = margin + random.random() * (width - 2*margin)
                y = margin + random.random() * (height - 2*margin)
                collision = False
                for p in self.points:
                    if ((p.x - x)**2 + (p.y - y)**2)**0.5 < min_dist:
                        collision = True
                        break
                if not collision:
                    self.points.append(Point(id=i, x=x, y=y))
                    valid_point = True
                    break
            if not valid_point:
                self.points.append(Point(id=i, x=x, y=y))
        
        center_x, center_y = width/2, height/2
        dists = [(p.id, (p.x-center_x)**2 + (p.y-center_y)**2) for p in self.points]
        self.depot_index = min(dists, key=lambda x: x[1])[0]
        
        self.edges = {i: [] for i in range(self.num_points)}
        coords = np.array([[p.x, p.y] for p in self.points])
        
        for i in range(self.num_points):
            dists_arr = np.sum((coords - coords[i])**2, axis=1)
            nearest = np.argsort(dists_arr)[1:self.k_neighbors+1]
            for neighbor in nearest:
                if neighbor not in self.edges[i]:
                    self.edges[i].append(int(neighbor))
                if i not in self.edges[int(neighbor)]:
                    self.edges[int(neighbor)].append(i)

        candidates = [i for i in range(self.num_points) if i != self.depot_index]
        random.shuffle(candidates)
        self.destinations = candidates[:self.num_destinations]
        self.visited_destinations = set()

    def regenerate(self, points, destinations):
        self.num_points = points
        self.num_destinations = destinations
        self.generate_map()
        self.status = "LOBBY"
        for aid, ag in self.agents.items():
            ag.idx = self.depot_index
        self.agents = {k:v for k,v in self.agents.items() if not v.is_cpu}

    def add_player(self, client_id: str):
        colors = ['#38bdf8', '#fbbf24', '#f472b6', '#a78bfa', '#34d399']
        color = random.choice(colors)
        if client_id not in self.agents:
            self.agents[client_id] = Agent(id=client_id, idx=self.depot_index, color=color)

    def add_cpus(self, count: int):
        self.agents = {k:v for k,v in self.agents.items() if not v.is_cpu}
        for i in range(count):
            cid = f"CPU_{i+1}"
            self.agents[cid] = Agent(id=cid, idx=self.depot_index, color="#94a3b8", is_cpu=True)

    def move_player(self, client_id: str, target_idx: int):
        if client_id in self.agents:
            self.agents[client_id].idx = target_idx
            if target_idx in self.destinations:
                self.visited_destinations.add(target_idx)
                return True
        return False
        
    def find_path(self, start_idx: int, end_idx: int) -> List[int]:
        if start_idx == end_idx: return [start_idx]
        pq = [(0, start_idx)]
        dists = {i: float('inf') for i in self.edges}
        dists[start_idx] = 0
        prev = {i: None for i in self.edges}
        while pq:
            d, u = heapq.heappop(pq)
            if d > dists[u]: continue
            if u == end_idx: break
            for v in self.edges[u]:
                p1 = self.points[u]; p2 = self.points[v]
                w = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5
                if dists[u] + w < dists[v]:
                    dists[v] = dists[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dists[v], v))
        path = []
        curr = end_idx
        if prev[curr] is None and curr != start_idx: return []
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        return path[::-1]

game = GameState()

# --- Connection Manager ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def broadcast(self, message: dict):
        txt = json.dumps(message)
        for connection in self.active_connections:
            try: await connection.send_text(txt)
            except: pass

manager = ConnectionManager()

# =============================================================================
# IMPROVED QUBO VRP SOLVER (Two-Phase: Assignment + TSP)
# =============================================================================

def get_euclidean_dist(idx1: int, idx2: int) -> float:
    """Get Euclidean distance between two point indices."""
    p1, p2 = game.points[idx1], game.points[idx2]
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5


def solve_assignment_qubo(targets: List[int], drivers: List[str]) -> Dict[str, List[int]]:
    """
    Phase 1: Assign each target to a driver using QUBO.
    Minimizes total distance from each driver's current position to assigned targets,
    while balancing the load.
    """
    num_targets = len(targets)
    num_drivers = len(drivers)
    
    if num_drivers == 0: return {}
    if num_targets == 0: return {d: [] for d in drivers}
    
    # If only one driver, assign all to them
    if num_drivers == 1:
        return {drivers[0]: targets[:]}
    
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    
    def var(k, i): return f"y_{k}_{i}"
    
    A = 10000.0  # Constraint weight (exactly one driver per target)
    B = 1.0      # Distance weight
    C = 500.0    # Load balancing weight
    
    # Constraint: Each target assigned to exactly one driver (One-Hot)
    for i in range(num_targets):
        vars_i = [var(k, i) for k in range(num_drivers)]
        # Sum = 1 constraint: -A * x + A * x_i * x_j (for i != j)
        for v in vars_i:
            bqm.add_variable(v, -A)
        for idx1 in range(len(vars_i)):
            for idx2 in range(idx1 + 1, len(vars_i)):
                bqm.add_interaction(vars_i[idx1], vars_i[idx2], 2 * A)
    
    # Objective: Minimize distance from driver's current pos to target
    for k in range(num_drivers):
        driver_pos = game.agents[drivers[k]].idx
        for i in range(num_targets):
            dist = get_euclidean_dist(driver_pos, targets[i])
            bqm.add_variable(var(k, i), dist * B)
    
    # Load Balancing: Penalize assigning too many targets to one driver
    # Encourage each driver to have ~ num_targets / num_drivers targets
    ideal_load = num_targets / num_drivers
    for k in range(num_drivers):
        vars_k = [var(k, i) for i in range(num_targets)]
        # Add quadratic penalty for having many targets (quadratic growth)
        for idx1 in range(len(vars_k)):
            for idx2 in range(idx1 + 1, len(vars_k)):
                bqm.add_interaction(vars_k[idx1], vars_k[idx2], C / num_targets)
    
    # Solve
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=5, num_sweeps=200)
    sample = response.first.sample
    
    # Parse result
    assignment = {d: [] for d in drivers}
    for k in range(num_drivers):
        for i in range(num_targets):
            if sample.get(var(k, i), 0) > 0:
                assignment[drivers[k]].append(targets[i])
    
    # Fallback: If any target not assigned, assign to nearest driver
    assigned_targets = set()
    for tlist in assignment.values():
        assigned_targets.update(tlist)
    
    for i, t in enumerate(targets):
        if t not in assigned_targets:
            # Find nearest driver
            best_k = 0
            best_dist = float('inf')
            for k, d in enumerate(drivers):
                dist = get_euclidean_dist(game.agents[d].idx, t)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k
            assignment[drivers[best_k]].append(t)
    
    return assignment


def solve_tsp_qubo_closed_loop(driver_id: str, targets: List[int]) -> List[int]:
    """
    Phase 2: Solve TSP for a single driver's assigned targets.
    Returns optimal visit order as a list of target indices, ending with depot.
    Uses Position-based TSP QUBO formulation for closed loop.
    """
    depot = game.depot_index
    start_pos = game.agents[driver_id].idx
    
    if len(targets) == 0:
        return [depot]
    
    if len(targets) == 1:
        return [targets[0], depot]
    
    # For small problems, use QUBO. For larger, use greedy.
    if len(targets) > 8:
        # Greedy nearest neighbor for performance
        route = []
        current = start_pos
        remaining = set(targets)
        while remaining:
            nearest = min(remaining, key=lambda t: get_euclidean_dist(current, t))
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
        route.append(depot)
        return route
    
    # ------- QUBO TSP (Position-based) -------
    n = len(targets)
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    
    def var(i, p): return f"z_{i}_{p}"
    
    A = 10000.0  # Constraint weight
    B = 1.0      # Distance weight
    
    # Constraint 1: Each target visited exactly once (row constraint)
    for i in range(n):
        vars_row = [var(i, p) for p in range(n)]
        for v in vars_row:
            bqm.add_variable(v, -A)
        for idx1 in range(len(vars_row)):
            for idx2 in range(idx1 + 1, len(vars_row)):
                bqm.add_interaction(vars_row[idx1], vars_row[idx2], 2 * A)
    
    # Constraint 2: Each position has exactly one target (column constraint)
    for p in range(n):
        vars_col = [var(i, p) for i in range(n)]
        for v in vars_col:
            bqm.add_variable(v, -A)
        for idx1 in range(len(vars_col)):
            for idx2 in range(idx1 + 1, len(vars_col)):
                bqm.add_interaction(vars_col[idx1], vars_col[idx2], 2 * A)
    
    # Objective: Minimize total distance
    # Distance: Start -> position 0
    for i in range(n):
        dist = get_euclidean_dist(start_pos, targets[i])
        bqm.add_variable(var(i, 0), dist * B)
    
    # Distance: position p -> position p+1
    for p in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dist = get_euclidean_dist(targets[i], targets[j])
                bqm.add_interaction(var(i, p), var(j, p + 1), dist * B)
    
    # Distance: position n-1 (last) -> depot (return home)
    for i in range(n):
        dist = get_euclidean_dist(targets[i], depot)
        bqm.add_variable(var(i, n - 1), dist * B)
    
    # Solve
    sampler = oj.SASampler()
    response = sampler.sample(bqm, num_reads=10, num_sweeps=500)
    sample = response.first.sample
    
    # Parse result: Find which target is at each position
    route = [None] * n
    for i in range(n):
        for p in range(n):
            if sample.get(var(i, p), 0) > 0:
                if route[p] is None:
                    route[p] = targets[i]
    
    # Fallback for missing positions
    assigned = set(r for r in route if r is not None)
    missing = [t for t in targets if t not in assigned]
    for p in range(n):
        if route[p] is None and missing:
            route[p] = missing.pop(0)
    
    # Remove any None and add depot
    route = [r for r in route if r is not None]
    route.append(depot)
    
    return route


def solve_vrp_two_phase(targets: List[int], drivers: List[str]) -> Dict[str, List[int]]:
    """
    Main VRP Solver: Two-Phase QUBO approach.
    Phase 1: Assign targets to drivers (Clustering QUBO)
    Phase 2: Solve TSP for each driver (Routing QUBO)
    """
    if not drivers:
        return {}
    if not targets:
        return {d: [game.depot_index] for d in drivers}
    
    print(f"  [VRP] Phase 1: Assigning {len(targets)} targets to {len(drivers)} drivers...")
    assignment = solve_assignment_qubo(targets, drivers)
    
    print(f"  [VRP] Phase 2: Solving TSP for each driver...")
    final_routes = {}
    for driver, assigned_targets in assignment.items():
        if assigned_targets:
            route = solve_tsp_qubo_closed_loop(driver, assigned_targets)
        else:
            route = [game.depot_index]
        final_routes[driver] = route
        print(f"    {driver}: {len(assigned_targets)} targets -> Route length {len(route)}")
    
    return final_routes


def solve_vrp_internally():
    """Called whenever routes need recalculation."""
    if game.status != "PLAYING":
        return {}
    
    targets = [d for d in game.destinations if d not in game.visited_destinations]
    if not targets:
        return {aid: [game.depot_index] for aid in game.agents}
    
    active_drivers = list(game.agents.keys())
    
    # Safety cap
    if len(targets) > 20:
        print("  [Warning] Too many targets, capping at 20.")
        targets = targets[:20]
    
    final_routes = solve_vrp_two_phase(targets, active_drivers)
    game.suggested_routes = final_routes
    return final_routes


# --- Background Task for CPU ---
async def cpu_simulation_loop():
    while True:
        await asyncio.sleep(2.0)
        if game.status == "PLAYING":
            cpus = [a for a in game.agents.values() if a.is_cpu]
            updates = []
            need_recalc = False
            
            for cpu in cpus:
                route = game.suggested_routes.get(cpu.id, [])
                if not route:
                    continue
                
                # Find next destination (skip visited ones and depot if not last)
                target_dest = None
                for r in route:
                    if r == game.depot_index:
                        # Only go to depot if it's the last remaining goal
                        remaining = [d for d in game.destinations if d not in game.visited_destinations]
                        if not remaining:
                            target_dest = r
                            break
                    elif r not in game.visited_destinations:
                        target_dest = r
                        break
                
                if target_dest is None:
                    continue
                
                if cpu.idx == target_dest:
                    # Arrived at destination
                    if target_dest in game.destinations:
                        game.visited_destinations.add(target_dest)
                        need_recalc = True
                    continue
                
                path = game.find_path(cpu.idx, target_dest)
                if len(path) > 1:
                    next_node = path[1]
                    old_idx = cpu.idx
                    game.agents[cpu.id].idx = next_node
                    
                    # Check if landed on a destination
                    if next_node in game.destinations and next_node not in game.visited_destinations:
                        game.visited_destinations.add(next_node)
                        need_recalc = True
                    
                    updates.append({"id": cpu.id, "idx": next_node})
            
            if updates or need_recalc:
                routes = solve_vrp_internally() if need_recalc else game.suggested_routes
                await manager.broadcast({
                    "type": "batch_update",
                    "updates": updates,
                    "visited": list(game.visited_destinations),
                    "routes": routes
                })

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cpu_simulation_loop())

# --- Websocket ---

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    print(f"New connection: {client_id}")
    await manager.connect(websocket)
    game.add_player(client_id)
    
    init_msg = {
        "type": "init",
        "status": game.status,
        "points": [p.dict() for p in game.points],
        "edges": game.edges,
        "destinations": game.destinations,
        "visited": list(game.visited_destinations),
        "depot": game.depot_index,
        "me_id": client_id,
        "players": [a.dict() for a in game.agents.values()]
    }
    await websocket.send_text(json.dumps(init_msg))
    
    await manager.broadcast({"type": "player_join", "player": game.agents[client_id].dict()})
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg["type"] == "start_game":
                cpu_count = msg.get("cpu_count", 0)
                game.add_cpus(cpu_count)
                game.status = "PLAYING"
                routes = solve_vrp_internally()
                await manager.broadcast({
                    "type": "game_start",
                    "players": [a.dict() for a in game.agents.values()],
                    "routes": routes
                })
                
            elif msg["type"] == "regenerate":
                p_count = int(msg.get("points", 80))
                d_count = int(msg.get("destinations", 8))
                game.regenerate(p_count, d_count) 
                await manager.broadcast({
                    "type": "init",
                    "status": game.status,
                    "points": [p.dict() for p in game.points],
                    "edges": game.edges,
                    "destinations": game.destinations,
                    "visited": [],
                    "depot": game.depot_index,
                    "players": [a.dict() for a in game.agents.values()]
                })

            elif msg["type"] == "move":
                if game.status == "PLAYING":
                    target = msg["target"]
                    game.move_player(client_id, target)
                    routes = solve_vrp_internally()
                    await manager.broadcast({
                        "type": "update_pos",
                        "id": client_id,
                        "idx": target,
                        "visited": list(game.visited_destinations),
                        "routes": routes
                    })
                
    except WebSocketDisconnect:
        print(f"Disconnected: {client_id}")
        manager.disconnect(websocket)
        if client_id in game.agents and not game.agents[client_id].is_cpu:
            del game.agents[client_id]
        await manager.broadcast({"type": "player_leave", "id": client_id})

# --- Static ---
static_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
