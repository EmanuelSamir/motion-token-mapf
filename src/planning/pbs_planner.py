import numpy as np
import heapq
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import deque
from src.planning.multi_phase_astar import MultiPhaseAStar
from src.planning.constraints import HighwayConstraints
from src.planning.motion_primitives import MotionPrimitives


class HighLevelNode:
    def __init__(
        self,
        constraints: Set[Tuple[int, int]],
        results: Dict[int, Dict[str, Any]],
        cost: float,
    ):
        self.constraints = constraints  # Set of (priority_v, lower_v)
        self.results = results  # v_id -> A* result dict
        self.cost = cost
        self.conflict = None

    def __lt__(self, other):
        return self.cost < other.cost


class PBSPlanner:
    """
    Implementation of Priority-Based Search (PBS) for Multi-Agent Planning.
    Based on the BK-PBS architecture.
    """

    def __init__(
        self,
        mp: MotionPrimitives,
        dt: float = 0.2,
        safe_radius: float = 7.0,
        margin_l: float = 0.20,
        margin_w: float = 0.20,
        horizon_agg: float = 150.0,
        horizon_safe: float = 80.0,
    ):
        self.mp = mp
        self.dt = dt
        self.astar = MultiPhaseAStar(
            mp, dt=dt, horizon_agg=horizon_agg, horizon_safe=horizon_safe
        )
        self.constraints_evaluator = HighwayConstraints(
            mp.road, safe_radius=safe_radius, margin_l=margin_l, margin_w=margin_w
        )
        self.agent_types = {}
        self.baseline_mode = "static"  # "static" or "constant_vel"

    def set_agent_types(self, agent_types: Dict[int, str]):
        self.agent_types = agent_types

    def set_baseline_mode(self, mode: str):
        self.baseline_mode = mode

    def get_priority_order(self, agents_states: Dict[int, np.ndarray]) -> List[int]:
        """Stable Priority Heuristic: Longitudinal Position + ID"""
        return sorted(
            agents_states.keys(),
            key=lambda vid: (agents_states[vid][0], vid),
            reverse=True,
        )

    def plan_all(
        self,
        agents_states: Dict[int, np.ndarray],
        v_info: List[Dict] = None,
        agents_history: Dict[int, List[List[float]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Entry point similar to PrioritizedPlanner.plan_all"""
        import time

        start_time = time.time()

        # 1. Update dimensions
        if v_info:
            dims = {v["id"]: (v["length"] / 2.0, v["width"] / 2.0) for v in v_info}
            self.constraints_evaluator.set_object_dimensions(dims)

        # 2. Initial Heuristic Order (Longitudinal)
        base_order = sorted(
            agents_states.keys(),
            key=lambda vid: (agents_states[vid][0], vid),
            reverse=True,
        )

        # 3. Root Node (Plan in initial order)
        root_results = self._plan_with_constraints(
            agents_states, set(), base_order=base_order, agents_history=agents_history
        )
        if not root_results:
            print("❌ [PBS] Critical: Root planning failed at priority 0.")
            return {}

        root = HighLevelNode(set(), root_results, self._calculate_cost(root_results))
        root.conflict = self._find_first_conflict(root.results)

        if not root.conflict:
            print("✅ [PBS] Initial plan is collision-free!")
            return root.results

        open_list = [root]

        # 3. PBS High-Level Search
        search_limit = 100
        steps = 0

        while open_list and steps < search_limit:
            steps += 1
            curr = heapq.heappop(open_list)

            # Find first branchable conflict
            conflict = self._find_first_conflict(curr.results, skip_t_zero=True)
            if not conflict:
                duration = time.time() - start_time
                print(
                    f"✅ [SUCCESS] PBS Solution found in {steps} iterations (Time: {duration:.3f}s)."
                )
                return curr.results

            v_i, v_j, t_crash = conflict

            # Avoid branching on the same pair if it's already constrained
            # (though the DAG should handle this, checking explicitly is safer)
            if (v_i, v_j) in curr.constraints or (v_j, v_i) in curr.constraints:
                # If we are here, it means the priority didn't solve the collision.
                # We skip this conflict to look for the next one in this node.
                continue

            print(
                f"🔀 [PBS] Iteration {steps}: Branching on Agent {v_i} vs {v_j} at T={t_crash * self.dt:.1f}s"
            )

            # Branch 1: v_i < v_j
            self._branch(
                curr,
                v_i,
                v_j,
                agents_states,
                open_list,
                target_conflict=(v_i, v_j),
                agents_history=agents_history,
            )

            # Branch 2: v_j < v_i
            self._branch(
                curr,
                v_j,
                v_i,
                agents_states,
                open_list,
                target_conflict=(v_i, v_j),
                agents_history=agents_history,
            )

        duration = time.time() - start_time
        print(
            f"🛑 [PBS] Reached limit ({search_limit}). Returning best-effort plan (Time: {duration:.3f}s)."
        )
        return curr.results

    def _branch(
        self,
        parent: HighLevelNode,
        high_v: int,
        low_v: int,
        agents_states,
        open_list,
        target_conflict=None,
        agents_history=None,
    ):
        new_constraints = parent.constraints.copy()
        new_constraints.add((high_v, low_v))

        if self._has_priority_cycle(new_constraints):
            return

        new_results = self._plan_with_constraints(
            agents_states,
            new_constraints,
            parent.results,
            affected_agent=low_v,
            agents_history=agents_history,
        )

        if new_results:
            # Verification: Did this actually solve the target conflict?
            if target_conflict:
                v_i, v_j = target_conflict
                if self._check_pair_collision(
                    v_i,
                    new_results[v_i]["path"],
                    v_j,
                    new_results[v_j]["path"],
                    skip_t_zero=True,
                ):
                    # It didn't solve it. We could still add it, but it might loop.
                    # In true PBS, we keep it, but here we'll add a cost penalty to prioritize other branches.
                    cost = self._calculate_cost(new_results) + 1000.0
                else:
                    cost = self._calculate_cost(new_results)
            else:
                cost = self._calculate_cost(new_results)

            new_node = HighLevelNode(new_constraints, new_results, cost)
            heapq.heappush(open_list, new_node)

    def _plan_with_constraints(
        self,
        agents_states: Dict[int, np.ndarray],
        constraints: Set[Tuple[int, int]],
        prev_results: Dict[int, Dict[str, Any]] = None,
        affected_agent: int = -1,
        base_order: List[int] = None,
        agents_history: Dict[int, List[List[float]]] = None,
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        Plans for agents following the partial order defined by constraints.
        """
        v_ids = list(agents_states.keys())
        if base_order is None:
            base_order = sorted(
                v_ids, key=lambda vid: (agents_states[vid][0], vid), reverse=True
            )

        order = self._get_topological_order(v_ids, constraints, base_order)

        results = {}
        planned_trajectories = {}

        for v_id in order:
            v_type = self.agent_types.get(v_id, "CAV")
            state = agents_states[v_id]

            if v_type == "CAV":
                if len(state) == 4:
                    lane_idx = self.mp.road.network.get_closest_lane_index(
                        state[:2], state[3]
                    )[2]
                    state = np.append(state, lane_idx)

                res = self.astar.plan(
                    state,
                    self.constraints_evaluator,
                    obstacles=planned_trajectories,
                    v_id=v_id,
                )

                if not res.get("path"):
                    return None  # Branch failure

                results[v_id] = res
                planned_trajectories[v_id] = res["path"]
            else:
                # BASELINE: Treat HDV with simplified non-reactive logic
                horizon = int(self.astar.horizon_agg)  # Approx horizon steps
                if len(state) == 4:
                    lane_idx = self.mp.road.network.get_closest_lane_index(
                        state[:2], state[3]
                    )[2]
                    state = np.append(state, lane_idx)

                if self.baseline_mode == "static":
                    # Option A: Stay at initial position
                    path = [state.copy() for _ in range(horizon)]
                else:
                    # Option B: Constant Velocity Linear Prediction
                    path = []
                    curr = state.copy()
                    dt = self.dt
                    vx = curr[2] * np.cos(curr[3])
                    vy = curr[2] * np.sin(curr[3])
                    for _ in range(horizon):
                        path.append(curr.copy())
                        curr[0] += vx * dt
                        curr[1] += vy * dt

                results[v_id] = {"path": path, "phase": "STATIC_BASELINE"}
                planned_trajectories[v_id] = path

        return results

    def _get_topological_order(
        self, v_ids: List[int], constraints: Set[Tuple[int, int]], base_order: List[int]
    ) -> List[int]:
        # Kahn's algorithm refined with base_order priority
        adj = {v: [] for v in v_ids}
        in_degree = {v: 0 for v in v_ids}

        for high, low in constraints:
            if high in adj and low in adj:
                adj[high].append(low)
                in_degree[low] += 1

        # Use a list for queue and sort it by base_order to maintain consistency
        queue = [v for v in v_ids if in_degree[v] == 0]
        # Sort queue so that agents with higher original priority are picked first if free
        priority_map = {vid: i for i, vid in enumerate(base_order)}
        queue.sort(key=lambda x: priority_map[x])

        order = []
        while queue:
            u = queue.pop(0)  # Pop highest priority agent with 0 in-degree
            order.append(u)

            # New nodes that become zero in-degree
            new_zeros = []
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    new_zeros.append(v)

            # Add to queue and re-sort by base_order
            queue.extend(new_zeros)
            queue.sort(key=lambda x: priority_map[x])

        return order

    def _has_priority_cycle(self, constraints: Set[Tuple[int, int]]) -> bool:
        # Check for cycles using DFS
        adj = {}
        nodes = set()
        for u, v in constraints:
            if u not in adj:
                adj[u] = []
            adj[u].append(v)
            nodes.add(u)
            nodes.add(v)

        visited = set()
        stack = set()

        def dfs(u):
            visited.add(u)
            stack.add(u)
            for v in adj.get(u, []):
                if v not in visited:
                    if dfs(v):
                        return True
                elif v in stack:
                    return True
            stack.remove(u)
            return False

        for n in nodes:
            if n not in visited:
                if dfs(n):
                    return True
        return False

    def _find_first_conflict(
        self, results: Dict[int, Dict[str, Any]], skip_t_zero: bool = False
    ) -> Optional[Tuple[int, int, int]]:
        """Finds the very first (v_i, v_j, t) conflict among all planned paths."""
        v_ids = list(results.keys())
        max_t = max([len(r["path"]) for r in results.values()])

        start_t = 1 if skip_t_zero else 0

        for t in range(start_t, max_t):
            for i in range(len(v_ids)):
                v_i = v_ids[i]
                path_i = results[v_i]["path"]
                if t >= len(path_i):
                    continue

                for j in range(i + 1, len(v_ids)):
                    v_j = v_ids[j]
                    path_j = results[v_j]["path"]
                    if t >= len(path_j):
                        continue

                    if self._is_colliding(v_i, path_i[t], v_j, path_j[t]):
                        return (v_i, v_j, t)
        return None

    def _check_pair_collision(
        self, id_a, path_a, id_b, path_b, skip_t_zero: bool = False
    ) -> bool:
        start_t = 1 if skip_t_zero else 0
        max_t = min(len(path_a), len(path_b))
        for t in range(start_t, max_t):
            if self._is_colliding(id_a, path_a[t], id_b, path_b[t]):
                return True
        return False

    def _is_colliding(self, id_a, s_a, id_b, s_j):
        # Quick circle check
        dist_sq = (s_a[0] - s_j[0]) ** 2 + (s_a[1] - s_j[1]) ** 2
        if dist_sq > 100:  # 10m radius
            return False

        # SAT check
        la, wa = self.constraints_evaluator.obj_dims.get(id_a, (2.5, 1.0))
        lb, wb = self.constraints_evaluator.obj_dims.get(id_b, (2.5, 1.0))

        # Add margin
        margin_l = self.constraints_evaluator.margin_l
        margin_w = self.constraints_evaluator.margin_w

        return self.constraints_evaluator._sat_overlap(
            s_a[0],
            s_a[1],
            s_a[3],
            s_j[0],
            s_j[1],
            s_j[3],
            lb + margin_l,
            wb + margin_w,
            hl1=la + margin_l,
            hw1=wa + margin_w,
        )

    def _calculate_cost(self, results: Dict[int, Dict[str, Any]]) -> float:
        # Sum of travel times (or total duration)
        return sum([len(r["path"]) for r in results.values()])
