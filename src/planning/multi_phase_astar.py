import numpy as np
import heapq
from typing import List, Dict, Any, Tuple, Optional
from src.planning.motion_primitives import MotionPrimitives


class AStarNode:
    _id_counter = 0

    def __init__(
        self,
        state: np.ndarray,
        t: int = 0,
        parent: "AStarNode" = None,
        action: str = None,
        g: float = 0.0,
        h: float = 0.0,
        history: List[str] = None,
    ):
        # state: [x, y, v, h, lane_idx]
        self.state = state
        self.t = t
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h
        self.f = g + h
        self.history = history or []
        self.node_id = AStarNode._id_counter
        AStarNode._id_counter += 1

    def __lt__(self, other):
        # Tie-break with H (favor nodes closer to goal)
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f


class MultiPhaseAStar:
    """
    BK-PBS Multi-phase A* with Phase and Control Telemetry Capture.
    """

    def __init__(
        self,
        primitives: MotionPrimitives,
        dt: float = 0.2,
        horizon_agg: float = 150.0,
        horizon_safe: float = 80.0,
    ):
        self.primitives = primitives
        self.dt = dt
        self.horizon_agg = horizon_agg
        self.horizon_safe = horizon_safe

    def plan(
        self,
        start_state: np.ndarray,
        constraints_evaluator,
        obstacles: Dict[int, List[np.ndarray]],
        max_x_limit: float = 9999.0,
        v_id: int = -1,
    ) -> Dict[str, Any]:
        """
        Executes multi-phase planning and returns a structured results dictionary.
        """
        results = {"all_phases": {}}

        # PHASE 1: Aggressive (Long-term goal)
        goal_agg = min(start_state[0] + self.horizon_agg, max_x_limit)
        p_agg, a_agg, t_agg, ids_agg, success_agg, r_agg = self.search(
            start_state,
            goal_agg,
            "aggressive",
            constraints_evaluator,
            obstacles,
            v_id=v_id,
        )

        results["all_phases"]["AGGRESSIVE"] = {
            "path": p_agg,
            "actions": a_agg,
            "tree": t_agg,
            "ids": ids_agg,
            "success": success_agg,
            "reason": r_agg,
        }

        if success_agg:
            results.update(
                {
                    "path": p_agg,
                    "actions": a_agg,
                    "tree_log": t_agg,
                    "chosen_ids": ids_agg,
                    "phase": "AGGRESSIVE",
                    "reason": r_agg,
                }
            )
            return results

        # PHASE 2: Safety (Emergency/Fall-back goal)
        goal_saf = min(start_state[0] + self.horizon_safe, max_x_limit)
        p_saf, a_saf, t_saf, ids_saf, success_saf, r_saf = self.search(
            start_state,
            goal_saf,
            "safety",
            constraints_evaluator,
            obstacles,
            v_id=v_id,
        )

        results["all_phases"]["SAFETY"] = {
            "path": p_saf,
            "actions": a_saf,
            "tree": t_saf,
            "ids": ids_saf,
            "success": success_saf,
            "reason": r_saf,
        }

        results.update(
            {
                "path": p_saf,
                "actions": a_saf,
                "tree_log": t_saf,
                "chosen_ids": ids_saf,
                "phase": "SAFETY",
                "reason": r_saf,
            }
        )
        return results

    def search(
        self,
        start_state: np.ndarray,
        goal_x: float,
        phase: str,
        constraints_evaluator,
        obstacles: Dict[int, List[np.ndarray]],
        v_id: int = -1,
    ) -> Tuple[
        Optional[List[np.ndarray]],
        Optional[List[str]],
        List[Dict],
        List[int],
        bool,
        str,
    ]:

        AStarNode._id_counter = 0
        budget = 5000  # Increased for better dense traffic handling
        v_ref = max(start_state[2], 15.0)
        h_val = (goal_x - start_state[0]) / v_ref

        root = AStarNode(start_state, 0, g=0.0, h=h_val, history=[])

        # ROOT VALIDATION: Ensure we don't start in collision or off-road
        start_on_road = constraints_evaluator._is_on_road(start_state, agent_id=v_id)
        start_collision_free = constraints_evaluator._is_collision_free(
            start_state, 0, obstacles, agent_id=v_id
        )

        if not (start_on_road and start_collision_free):
            reason = (
                "INITIAL_COLLISION" if not start_collision_free else "INITIAL_OFF_ROAD"
            )
            print(
                f"❌ [A* ERROR] Agent {v_id} starts in invalid state: {reason} at {start_state[:2]}"
            )
            return None, None, [], [], False, reason

        open_list = [root]
        closed_set = {}
        best_node = root

        # Tree Capture: Initialize with Root
        tree_log = [
            {
                "parent_id": None,
                "node_id": root.node_id,
                "action": "START",
                "state": root.state.copy(),
                "f": root.f,
                "g": root.g,
                "h": root.h,
                "valid": True,
                "lane_type": self._get_lane_label(root.state),
                "acc": 0.0,
                "steer": 0.0,
            }
        ]

        all_children = []
        step_count = 0

        while open_list and step_count < budget:
            step_count += 1
            curr = heapq.heappop(open_list)

            if curr.state[0] >= goal_x:
                p, a, ids = self.reconstruct_path(curr)
                return p, a, tree_log, ids, True, "SUCCESS"
                return p, a, tree_log, ids, True, "SUCCESS"

            if curr.state[0] > best_node.state[0]:
                best_node = curr

            key = (int(curr.state[0]), int(curr.state[4]), curr.t)
            if key in closed_set and closed_set[key] <= curr.g:
                continue
            closed_set[key] = curr.g

            maneuvers = self.primitives.get_maneuvers(curr.state, phase)
            if phase == "safety":
                maneuvers = [
                    m for m in maneuvers if m not in ["ACCELERATE"]
                ]  # , "IDLE"]]

            for m in maneuvers:
                # Capture Low Level Control for Telemetry
                control = self.primitives.get_low_level_action(curr.state, m)

                segment = self.primitives.propagate(curr.state, m, self.dt)

                # VALIDATION: Check ALL sub-steps of the maneuver, not just the end state
                is_valid = True
                fail_type = "NONE"

                # curr.t is the starting sub-step index
                if not constraints_evaluator.is_valid(
                    segment, curr.t, obstacles, agent_id=v_id
                ):
                    is_valid = False
                    fail_type = "COLLISION"

                future_state = segment[-1]
                on_road = constraints_evaluator._is_on_road(future_state, agent_id=v_id)
                if is_valid and not on_road:
                    is_valid = False
                    fail_type = "OFF_ROAD"

                # Log all nodes (including rejected ones) for full tree visualization
                next_state = future_state
                node_id_to_log = AStarNode._id_counter
                AStarNode._id_counter += 1  # Pre-allocate ID for rejected nodes too

                tree_log.append(
                    {
                        "parent_id": curr.node_id,
                        "node_id": node_id_to_log,
                        "action": m,
                        "state": next_state.copy(),
                        "g": curr.g + self.dt,
                        "h": (goal_x - next_state[0]) / max(next_state[2], 1.0),
                        "f": (curr.g + self.dt)
                        + ((goal_x - next_state[0]) / max(next_state[2], 1.0)),
                        "valid": is_valid,
                        "fail_type": fail_type,
                        "lane_type": self._get_lane_label(next_state),
                        "acc": control["acceleration"],
                        "steer": control["steering"],
                    }
                )
                all_children.append(
                    (curr, None, is_valid, control, m)
                )  # Simplified for rejection tracking

                # Ensure all ancestors are in the log (needed for full tree traces)
                trace_ptr = curr
                while trace_ptr and trace_ptr.parent:
                    if any(n["node_id"] == trace_ptr.node_id for n in tree_log):
                        break
                    tree_log.append(
                        {
                            "parent_id": trace_ptr.parent.node_id,
                            "node_id": trace_ptr.node_id,
                            "action": trace_ptr.action,
                            "state": trace_ptr.state.copy(),
                            "valid": True,
                            "lane_type": self._get_lane_label(trace_ptr.state),
                            "g": trace_ptr.g,
                            "h": trace_ptr.h,
                            "f": trace_ptr.f,
                            "acc": 0.0,
                            "steer": 0.0,
                        }
                    )
                    trace_ptr = trace_ptr.parent

                if not is_valid:
                    continue
                next_node = AStarNode(
                    next_state,
                    curr.t + len(segment),
                    curr,
                    m,
                    g=curr.g + self.dt,
                    h=(goal_x - next_state[0]) / max(next_state[2], 1.0),
                    history=curr.history + [m],
                )
                next_node.node_id = node_id_to_log
                heapq.heappush(open_list, next_node)

        reason = "BUDGET_EXCEEDED" if step_count >= budget else "BLOCKED_BY_COLLISION"
        if not all_children:
            reason = "NO_VALID_MANEUVERS"

        res_path, res_actions, chosen_ids = self.reconstruct_path(best_node)
        res_path, res_actions, chosen_ids = self.reconstruct_path(best_node)

        if phase == "safety" and not res_actions:
            return (
                [start_state],
                ["HARD_BRAKE"],
                tree_log,
                [root.node_id],
                False,
                "EMERGENCY_BRAKE",
            )
        return res_path, res_actions, tree_log, chosen_ids, False, reason

    def _get_lane_label(self, state: np.ndarray) -> str:
        if not self.primitives.road or len(state) < 5:
            return "HWY"
        x, y, _, h, _ = state
        lane_index = self.primitives.road.network.get_closest_lane_index(
            np.array([x, y]), h
        )
        if lane_index[0] in ["j", "k"]:
            return "RAMP"
        return "HWY"

    def reconstruct_path(
        self, node: AStarNode
    ) -> Tuple[List[np.ndarray], List[str], List[int]]:
        if not node or not node.parent:
            return None, None, []
        p, a, ids = [], [], []
        curr = node
        while curr and curr.parent:
            p.append(curr.state)
            a.append(curr.action)
            ids.append(curr.node_id)
            curr = curr.parent
        if curr:
            p.append(curr.state)
            ids.append(curr.node_id)
        return p[::-1], a[::-1], ids[::-1]
