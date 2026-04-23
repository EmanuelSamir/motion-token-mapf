import numpy as np
from typing import List, Dict, Any, Tuple


class MotionPrimitives:
    """
    Zero-Discrepancy Motion Primitives.
    An exact port of highway-env's cascaded controller and bicycle kinematics.
    """

    def __init__(self, road=None, vehicle_params: Dict[str, float] = None):
        self.road = road
        self.params = vehicle_params or {"length": 5.0, "width": 2.0}
        self.LENGTH = self.params.get("length", 5.0)
        self.WIDTH = self.params.get("width", 2.0)
        self.MAX_SPEED = 40.0
        self.MIN_SPEED = -40.0

        # Exact constants from highway_env.vehicle.controller.ControlledVehicle
        self.TAU_ACC = 0.6
        self.TAU_HEADING = 0.2
        self.TAU_LATERAL = 0.6
        self.TAU_PURSUIT = 0.5 * self.TAU_HEADING  # 0.1s
        self.KP_HEADING = 1 / self.TAU_HEADING  # 5.0
        self.KP_LATERAL = 1 / self.TAU_LATERAL  # 1.666
        self.MAX_STEERING_ANGLE = np.pi / 3

        # Maneuvers based on Section IV-A
        self.MANEUVERS = {
            "IDLE": {"acc": 0.0, "lane_change": 0},
            "ACCELERATE": {"acc": 5.0, "lane_change": 0},
            "DECELERATE": {"acc": -3.0, "lane_change": 0},
            "HARD_BRAKE": {"acc": -8.0, "lane_change": 0},
            "LANE_CHANGE_LEFT": {"acc": 0.0, "lane_change": -1},
            "LANE_CHANGE_RIGHT": {"acc": 0.0, "lane_change": 1},
        }

    def _wrap_to_pi(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _not_zero(self, x, eps=1e-2):
        if abs(x) > eps:
            return x
        return eps if x >= 0 else -eps

    def _bicycle_step(
        self,
        x: float,
        y: float,
        v: float,
        h: float,
        acc: float,
        steer: float,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        # High-Fidelity Sync with highway-env integration order:
        # 1. Heading first (using old V)
        new_h = h + v * np.tan(steer) / self.LENGTH * dt

        # 2. Position second (using old V, but NEW heading)
        new_x = x + v * np.cos(new_h) * dt
        new_y = y + v * np.sin(new_h) * dt

        # 3. Velocity last
        new_v = v + acc * dt
        new_v = np.clip(new_v, self.MIN_SPEED, self.MAX_SPEED)

        return new_x, new_y, new_v, new_h

    def get_maneuvers(self, state: np.ndarray, phase: str = "aggressive") -> List[str]:
        if phase == "aggressive":
            base = ["ACCELERATE", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "IDLE"]
        else:  # safety
            base = [
                "DECELERATE",
                "HARD_BRAKE",
                "LANE_CHANGE_LEFT",
                "LANE_CHANGE_RIGHT",
                "IDLE",
            ]

        if not self.road:
            return base
        x, y, v, h, curr_l = state
        lane_index = self.road.network.get_closest_lane_index(np.array([x, y]), h)
        side_lanes = self.road.network.all_side_lanes(lane_index)
        curr_idx = lane_index[2]

        filtered = []
        for m_name in base:
            m = self.MANEUVERS[m_name]
            if m["lane_change"] == -1 and curr_idx <= 0:
                continue
            if m["lane_change"] == 1 and curr_idx >= len(side_lanes) - 1:
                continue
            filtered.append(m_name)
        return filtered

    def propagate(
        self, start_state: np.ndarray, maneuver: str, dt: float
    ) -> List[np.ndarray]:
        curr_x, curr_y, curr_v, curr_h, curr_l = start_state
        traj = []

        # SYNC: Match 15Hz simulation frequency (3 steps of 0.066s per 0.2s policy step)
        num_substeps = 3
        dt_sub = dt / num_substeps

        # SYNC: Constant control for the duration of the policy step (fidelity match)
        action = self.get_low_level_action(start_state, maneuver)

        for _ in range(num_substeps):
            curr_x, curr_y, curr_v, curr_h = self._bicycle_step(
                curr_x,
                curr_y,
                curr_v,
                curr_h,
                action["acceleration"],
                action["steering"],
                dt_sub,
            )
            # Virtual lane index maintenance
            curr_l = action["target_lane_idx"]
            traj.append(np.array([curr_x, curr_y, curr_v, curr_h, curr_l]))
        return traj

    def get_low_level_action(
        self, state: np.ndarray, maneuver: str
    ) -> Dict[str, float]:
        """
        TRANSPLANT: Verbatim copy of ControlledVehicle.steering_control
        from highway_env/vehicle/controller.py
        """
        x, y, v, h, curr_l = state
        m = self.MANEUVERS.get(maneuver, self.MANEUVERS["IDLE"])
        acceleration = m["acc"]

        if self.road:
            # 1. Geometric Lane Identification (matches simulator's logic)
            anchor_lane_index = self.road.network.get_closest_lane_index(
                np.array([x, y]), h
            )
            # Use the actual closest lane instead of state[4] to prevent divergence at segments
            curr_l_geometric = int(anchor_lane_index[2])
            target_idx = int(curr_l_geometric + m["lane_change"])
            target_idx = max(
                0,
                min(
                    target_idx,
                    len(self.road.network.all_side_lanes(anchor_lane_index)) - 1,
                ),
            )

            final_lane_idx = (anchor_lane_index[0], anchor_lane_index[1], target_idx)
            target_lane = self.road.network.get_lane(final_lane_idx)

            # --- START LIBRARY STEERING CONTROL ---
            lane_coords = target_lane.local_coordinates(np.array([x, y]))
            # Pursuit look-ahead
            lane_next_coords = lane_coords[0] + v * self.TAU_PURSUIT
            lane_future_heading = target_lane.heading_at(lane_next_coords)

            # Lateral position control (P)
            lateral_speed_command = -self.KP_LATERAL * lane_coords[1]

            # Lateral speed to heading
            heading_command = np.arcsin(
                np.clip(lateral_speed_command / self._not_zero(v), -1, 1)
            )
            heading_ref = lane_future_heading + np.clip(
                heading_command, -np.pi / 4, np.pi / 4
            )

            # Heading control (P)
            heading_rate_command = self.KP_HEADING * self._wrap_to_pi(heading_ref - h)

            # Heading rate to steering angle
            slip_angle = np.arcsin(
                np.clip(
                    self.LENGTH / 2 / self._not_zero(v) * heading_rate_command, -1, 1
                )
            )
            steering = np.arctan(2 * np.tan(slip_angle))
            # --- END LIBRARY STEERING CONTROL ---

            target_lane_idx = target_idx
        else:
            steering = 0.0
            target_lane_idx = curr_l

        return {
            "acceleration": acceleration,
            "steering": np.clip(
                steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
            ),
            "target_lane_idx": target_lane_idx,
        }
