
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
        
        # Reduced default count for "Full QUBO" performance
        self.num_points = 80
        self.num_destinations = 8  # Kept small for Full VRP QUBO real-time solving
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
            dists = np.sum((coords - coords[i])**2, axis=1)
            nearest = np.argsort(dists)[1:self.k_neighbors+1]
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
        self.num_destinations = 8 # Force small number for VRP safety
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

# --- FULL VRP QUBO Solver ---

def solve_full_vrp_qubo(targets: List[int], active_drivers: List[str]) -> Dict[str, List[int]]:
    """
    Solves Multi-Agent VRP completely using QUBO.
    Decides BOTH assignment (clustering) AND routing (TSP) in one giant BQM.
    """
    num_drivers = len(active_drivers)
    num_targets = len(targets)
    if num_drivers == 0 or num_targets == 0: return {}

    print(f"  [QUBO VRP] Solving for {num_drivers} agents, {num_targets} targets...")
    
    # 1. Prepare Nodes Mapping
    # Logic: Each agent has T steps.
    # We fix step 0 to be the current location of the agent.
    # We optimize from step 1 to num_targets.
    # To save variables, we assume total steps = num_targets (worst case one agent does all).
    # But usually we can cap steps at ceil(num_targets / num_drivers) + buffer.
    
    max_steps_per_agent = max(2, int(num_targets * 0.8) + 1)
    # Total limit for safety. If variables > ~2000 it gets too slow.
    # Variables ~ Agents * Steps * Targets
    
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    
    # Coordinates for penalty calculation
    target_points = {t: game.points[t] for t in targets}
    agent_starts = {aid: game.points[game.agents[aid].idx] for aid in active_drivers}
    
    # Precompute distances
    # dist_matrix[u][v]
    all_indices = targets + [game.agents[aid].idx for aid in active_drivers] + [game.depot_index]
    all_indices = list(set(all_indices))
    
    def get_dist(i, j):
        p1 = game.points[i]; p2 = game.points[j]
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

    # Variables: x_{k, t, i} -> Agent k visits Target i at Step t
    # k: 0..num_drivers-1
    # t: 0..max_steps_per_agent-1 (0 is first move from start)
    # i: 0..num_targets-1 (index in targets list)
    
    def var(k, t, i): return f"x_{k}_{t}_{i}"
    
    A = 5000.0  # Constraint Penalty (High)
    B = 1.0     # Distance Weight (Low)
    
    # Constraint 1: Every target i must be visited exactly once by ANY agent at ANY step
    for i in range(num_targets):
        # sum_{k,t} x_{k,t,i} == 1
        # Terms: -A * x + 2A * pair
        vars_for_i = []
        for k in range(num_drivers):
            for t in range(max_steps_per_agent):
                v = var(k, t, i)
                vars_for_i.append(v)
                bqm.add_variable(v, -A)
        
        for idx1 in range(len(vars_for_i)):
            for idx2 in range(idx1 + 1, len(vars_for_i)):
                bqm.add_interaction(vars_for_i[idx1], vars_for_i[idx2], 2 * A)

    # Constraint 2: Each agent k at step t can visit at most one target
    for k in range(num_drivers):
        for t in range(max_steps_per_agent):
            # sum_{i} x_{k,t,i} <= 1
            # We use a slack variable or penalty for >1. 
            # Ideally = 1 if we forced full coverage, but maybe they stop early?
            # Actually, let's allow "No Move" (dummy node)?
            # To simplify, we enforce: sum_{i} x_{k,t,i} <= 1. 
            # Penalty: sum(pairs) * 2A. No linear term change?
            # Incorrect. (sum x)^2 = sum x^2 + pairs. 
            # If we want sum <= 1, we penalize pairs only.
            
            vars_for_step = [var(k, t, i) for i in range(num_targets)]
            for idx1 in range(len(vars_for_step)):
                for idx2 in range(idx1 + 1, len(vars_for_step)):
                    bqm.add_interaction(vars_for_step[idx1], vars_for_step[idx2], 2 * A)

    # Objective: Minimize Distance
    for k in range(num_drivers):
        aid = active_drivers[k]
        start_node = game.agents[aid].idx
        
        # Step 0: From Start -> Target i
        for i in range(num_targets):
            dist = get_dist(start_node, targets[i])
            bqm.add_variable(var(k, 0, i), dist * B)
            
        # Step t -> t+1: Target i -> Target j
        for t in range(max_steps_per_agent - 1):
            for i in range(num_targets):
                for j in range(num_targets):
                    if i == j: continue
                    dist = get_dist(targets[i], targets[j])
                    # Interaction: if x_{k,t,i} AND x_{k,t+1,j} are 1, add dist
                    bqm.add_interaction(var(k, t, i), var(k, t+1, j), dist * B)

    # Solve
    try:
        sampler = oj.SASampler()
        start_time = time.time()
        response = sampler.sample(bqm, num_reads=1, num_sweeps=100) # Fast anneal
        print(f"  [QUBO VRP] Optimized in {time.time() - start_time:.3f}s")
        
        sample = response.first.sample
        
        # Decode
        final_routes = {aid: [] for aid in active_drivers}
        
        for k in range(num_drivers):
            aid = active_drivers[k]
            # Extract sequence (t, i)
            steps = []
            for key, val in sample.items():
                if val > 0 and key.startswith(f"x_{k}_"):
                    _, k_str, t_str, i_str = key.split('_')
                    steps.append((int(t_str), int(i_str)))
            
            # Sort by step
            steps.sort(key=lambda x: x[0])
            
            # Convert to target IDs
            route = []
            for _, i_idx in steps:
                if i_idx < len(targets):
                    route.append(targets[i_idx])
            
            # Append return to depot
            route.append(game.depot_index)
            final_routes[aid] = route
            
        return final_routes
        
    except Exception as e:
        print(f"  [QUBO Error] {e}")
        return {aid: [game.depot_index] for aid in active_drivers}


def solve_vrp_internally():
    if game.status != "PLAYING": return {}
    targets = [d for d in game.destinations if d not in game.visited_destinations]
    if not targets: return {aid: [game.depot_index] for aid in game.agents}
    
    active_drivers = list(game.agents.keys())
    
    # Use Full QUBO Solver
    # Note: If target count is huge, this might lag.
    if len(targets) > 12:
        # Emergency fallback to slicing simply to prevent crash, 
        # but user asked for QUBO, so we try our best or slice.
        print("  [Warning] Targets > 12. Slicing for safety.")
        targets = targets[:12]

    final_routes = solve_full_vrp_qubo(targets, active_drivers)
    game.suggested_routes = final_routes
    return final_routes

# --- Background Task for CPU ---
async def cpu_simulation_loop():
    while True:
        await asyncio.sleep(2.5) # Slower movement
        if game.status == "PLAYING":
            cpus = [a for a in game.agents.values() if a.is_cpu]
            updates = []
            
            for cpu in cpus:
                route = game.suggested_routes.get(cpu.id, [])
                if route:
                    target_dest = route[0]
                    if cpu.idx == target_dest: continue
                    path = game.find_path(cpu.idx, target_dest)
                    if len(path) > 1:
                        next_node = path[1]
                        game.move_player(cpu.id, next_node)
                        updates.append({"id": cpu.id, "idx": next_node})
                    
            if updates:
                # Re-solve VRP completely on every move
                routes = solve_vrp_internally()
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
                p_count = int(msg.get("points", 100))
                # FORCE 8 destinations for QUBO stability
                game.regenerate(p_count, 8) 
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
