import numpy as np
from typing import List, Dict, Any, Tuple
from src.planning.multi_phase_astar import MultiPhaseAStar
from src.planning.motion_primitives import MotionPrimitives
from src.planning.constraints import HighwayConstraints


class PrioritizedPlanner:
    """
    Coordinates multiple CAVs using a clean sequential prioritized planning approach.
    Uses generic A* search with modular highway-specific constraints.
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

    def get_priority_order(self, agents_states: Dict[int, np.ndarray]) -> List[int]:
        """Stable Priority Heuristic: Longitudinal Position + ID (to ensure deterministic tie-breaking)"""
        return sorted(
            agents_states.keys(),
            key=lambda vid: (agents_states[vid][0], vid),
            reverse=True,
        )

    def plan_all(
        self,
        agents_states: Dict[int, np.ndarray],
        other_obstacles: Dict[int, List[np.ndarray]] = None,
        v_info: List[Dict] = None,
        max_x_limit: float = 9999.0,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Sequential planning by priority with randomization retry.
        """
        if not agents_states:
            return {}

        # 1. Update Dimensions in Evaluator for precise SAT
        if v_info:
            dims = {v["id"]: (v["length"] / 2.0, v["width"] / 2.0) for v in v_info}
            self.constraints_evaluator.set_object_dimensions(dims)

        # Stable Priority Heuristic: Longitudinal Position + ID (to ensure deterministic tie-breaking)
        base_sorted_ids = self.get_priority_order(agents_states)

        max_retries = 10
        for attempt in range(max_retries):
            if attempt > 0:
                print(
                    f"🔄 [RETRY] Randomizing priorities (Attempt {attempt + 1}/{max_retries})..."
                )
                import random

                random.shuffle(base_sorted_ids)
            else:
                print(
                    f"🚀 [PLAN] Starting initial planning attempt with priority: {base_sorted_ids}"
                )

            all_planned_trajectories = {}
            if other_obstacles:
                all_planned_trajectories.update(other_obstacles)

            # RESULTS DICTIONARY
            results = {}
            success = True

            for v_id in base_sorted_ids:
                state = agents_states[v_id]
                # Ensure state is [x, y, v, h, lane_idx]
                if len(state) == 4:
                    lane_idx = self.mp.road.network.get_closest_lane_index(
                        state[:2], state[3]
                    )[2]
                    state = np.append(state, lane_idx)

                res = self.astar.plan(
                    state,
                    self.constraints_evaluator,
                    obstacles=all_planned_trajectories,
                    max_x_limit=max_x_limit,
                    v_id=v_id,
                )

                path = res.get("path")
                v_path_valid = True
                if path:
                    # [DEBUG] Sub-step validation of the chosen path
                    for t_idx, s in enumerate(path):
                        # Node index t_idx matches sub-step t_idx*3
                        sub_step_t = t_idx * 3
                        if not self.constraints_evaluator._is_collision_free(
                            s, sub_step_t, all_planned_trajectories, v_id
                        ):
                            print(
                                f"🔍 [FORENSIC] Path for {v_id} rejected: sub-step collision at node {t_idx} (sub-step {sub_step_t})!"
                            )
                            v_path_valid = False
                            break

                if path and v_path_valid:
                    results[v_id] = res
                    all_planned_trajectories[v_id] = path
                else:
                    reason = "NO_PATH" if not path else "FORENSIC_COLLISION"
                    print(
                        f"❌ [FAIL] Agent {v_id} failed in attempt {attempt + 1} ({reason}). Retrying..."
                    )
                    results[v_id] = {
                        "path": [],
                        "actions": [],
                        "phase": "FAILED",
                        "reason": reason,
                    }
                    success = False
                    break

            if success:
                return results

        return results
