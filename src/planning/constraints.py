import numpy as np
from typing import List, Dict, Tuple, Optional


class HighwayConstraints:
    """
    Centralized constraint evaluator for highway coordination.
    Pure Paper Implementation: Bounding Box Overlap.
    """

    def __init__(
        self,
        road,
        half_l: float = 2.5,
        half_w: float = 1.0,
        safe_radius: float = 7.0,
        margin_l: float = 0.20,
        margin_w: float = 0.20,
    ):
        self.road = road
        self.margin_l = margin_l
        self.margin_w = margin_w
        # Safety margins (added to base half-dimensions)
        self.agent_hl = half_l + margin_l
        self.agent_hw = half_w + margin_w
        self.obj_dims = {}  # Map of v_id -> (hl, hw)
        self.safe_radius_sq = safe_radius**2
        # Restored to 10.0 to allow for the physical gap between Ramp and Highway during merges
        self.MAX_LAT_OFFSET = 10.0

    def set_object_dimensions(self, dims: Dict[int, Tuple[float, float]]):
        """Standardize dimensions for SAT checks."""
        self.obj_dims = dims  # id -> (hl, hw)

    def is_valid(
        self,
        segment: List[np.ndarray],
        start_t: int,
        obstacles: Dict[int, List[np.ndarray]],
        agent_id: int = -1,
    ) -> bool:
        for i, state in enumerate(segment):
            t = start_t + i + 1
            if not self._is_on_road(state, agent_id):
                return False
            if not self._is_collision_free(state, t, obstacles, agent_id):
                return False
        return True

    def _is_on_road(self, state: np.ndarray, agent_id: int = -1) -> bool:
        if not self.road:
            return True
        x, y, _, h, state_lane_idx = state
        lane_index = self.road.network.get_closest_lane_index(np.array([x, y]), h)

        # Paper Check: Are we in the lane set? (Current and Adjacent)
        # We allow a loose check to prevent pruning valid transition branches
        # Raised to 3.0 to allow Ramp (0) to Highway (2) transitions
        if abs(int(lane_index[2]) - int(state_lane_idx)) > 3.0:
            print(
                f"[DEBUG OFF_ROAD] Agent {agent_id}: Index Mismatch! LaneIdx:{lane_index[2]} vs State:{state_lane_idx} | Road:{lane_index[0]} | x:{x:.1f}"
            )
            return False

        lane = self.road.network.get_lane(lane_index)
        _, lat = lane.local_coordinates(np.array([x, y]))

        is_on = abs(lat) <= self.MAX_LAT_OFFSET
        if not is_on:
            print(
                f"[DEBUG OFF_ROAD] Agent {agent_id}: Lat Offset {lat:.2f} > {self.MAX_LAT_OFFSET} | x:{x:.1f}, y:{y:.1f}"
            )
        return is_on

    def _is_collision_free(
        self,
        state: np.ndarray,
        t: int,
        obstacles: Dict[int, List[np.ndarray]],
        agent_id: int = -1,
        verbose: bool = False,
    ) -> bool:
        """
        Check collision against:
        1. Dynamic trajectories in 'obstacles'
        2. Static objects in 'self.road.objects'
        """
        x_a, y_a, _, h_a, _ = state

        # --- 1. DYNAMIC OBSTACLES (Vehicles) ---
        for obs_id, traj in obstacles.items():
            if obs_id == agent_id:
                continue

            # Direct temporal mapping: Map sub-step index (15Hz) to search node index (5Hz)
            # k = t // 3 ensures we align 3 sub-steps to 1 search node.
            k = min(t // 3, len(traj) - 1)
            obs = traj[k]
            x_b, y_b, h_b = obs[0], obs[1], obs[3]

            if (x_a - x_b) ** 2 + (y_a - y_b) ** 2 > self.safe_radius_sq:
                continue

            hl_b, hw_b = self.obj_dims.get(obs_id, (2.5, 1.0))
            if self._sat_overlap(x_a, y_a, h_a, x_b, y_b, h_b, hl_b, hw_b, verbose=verbose):
                return False

        # --- 2. STATIC OBSTACLES (Road Objects) ---
        if hasattr(self.road, "objects") and self.road.objects:
            for obj in self.road.objects:
                x_b, y_b, h_b = obj.position[0], obj.position[1], obj.heading
                hl_b = getattr(obj, "length", getattr(obj, "LENGTH", 2.0)) / 2.0
                hw_b = getattr(obj, "width", getattr(obj, "WIDTH", 2.0)) / 2.0

                if (x_a - x_b) ** 2 + (y_a - y_b) ** 2 > self.safe_radius_sq:
                    continue

                if self._sat_overlap(x_a, y_a, h_a, x_b, y_b, h_b, hl_b, hw_b, verbose=verbose):
                    return False

        return True

    def _sat_overlap(self, x1, y1, h1, x2, y2, h2, hl2, hw2, hl1=None, hw1=None, verbose=False) -> bool:
        hl1 = hl1 if hl1 is not None else self.agent_hl
        hw1 = hw1 if hw1 is not None else self.agent_hw
        
        c1, s1 = np.cos(h1), np.sin(h1)
        c2, s2 = np.cos(h2), np.sin(h2)
        dx, dy = x1 - x2, y1 - y2

        # SAT axes (normals of both boxes)
        for i, (ax, ay) in enumerate([(c1, s1), (-s1, c1), (c2, s2), (-s2, c2)]):
            d_proj = abs(dx * ax + dy * ay)
            # Projection of box 1
            r1 = hl1 * abs(c1 * ax + s1 * ay) + hw1 * abs(-s1 * ax + c1 * ay)
            # Projection of box 2
            r2 = hl2 * abs(c2 * ax + s2 * ay) + hw2 * abs(-s2 * ax + c2 * ay)

            if d_proj > r1 + r2:
                return False

            if verbose:
                print(f"      [SAT axis {i}] overlap: {(r1+r2)-d_proj:.3f}m | d_proj:{d_proj:.3f} | r1+r2:{r1+r2:.3f}")

        return True
