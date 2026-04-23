import numpy as np
import gymnasium as gym
from typing import Tuple, List, Dict, Any, Optional

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


class CAVVehicle(IDMVehicle):
    """
    A specialized IDMVehicle that can be externally overridden.
    Its manual control is 'sticky' within a simulation step to survive internal act(None) calls.
    """

    def act(self, action: dict | str = None):
        if action is not None and isinstance(action, dict):
            # Bypass IDM and use external planned action
            Vehicle.act(self, action)
            self._external_controlled = True
            self._manual_action_applied = True  # Flag to survive the road.step() call
        elif getattr(self, "_manual_action_applied", False):
            # Already acted this step via planner, ignore the fallback act(None)
            pass
        else:
            # Fall back to Intelligent Driver Model
            self._external_controlled = False
            super().act(action)


class MergeInteractionEnv(AbstractEnv):
    """
    A specialized highway merge environment for CAV and HDV interaction.
    Features:
    - Parametrizable highway and ramp lanes.
    - Infinite flow management (count & ratio maintenance).
    - Randomized driver behaviors (aggressiveness, speeds).
    - Mixed autonomy support (CAV/HDV ratio).
    - Fixed camera perspective.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": True,
                    "order": "sorted",
                },
                "action": {
                    "type": "ContinuousAction",
                },
                # Road Parameters
                "highway_lanes": 2,
                "ramp_lanes": 1,
                "ramp_distance": 150,  # Distance before merging
                "merging_length": 80,  # Length of the merging zone
                "post_merge_length": 200,  # Length after merging zone
                "ramp_curve_length": 50,
                "ramp_offset": 2.0,
                "duration": 200,
                "lane_width": StraightLane.DEFAULT_WIDTH,
                # Vehicle Parameters
                "vehicle_length": 5.0,
                "vehicle_width": 2.0,
                "target_vehicles_count": 30,
                "cav_ratio": 0.3,
                "spawn_gap": 25.0,  # Minimum distance between spawns
                "initial_speed_range": [20, 25],
                "target_speed_range": [30, 35],
                # Granular Randomization Parameters
                "time_wanted_range": [0.8, 2.0],
                "politeness_range": [0.1, 0.7],
                "distance_wanted_range": [5.0, 15.0],
                "min_acc_gain_range": [0.1, 0.4],
                "acc_max_range": [4.0, 6.0],
                # Simulation Parameters
                "policy_frequency": 5,
                "simulation_frequency": 15,
                "screen_width": 1200,
                "screen_height": 400,
                "centering_position": [0.5, 0.5],
                "scaling": 3.0,
                "fixed_camera": True,
                "roi_center": 220,  # Center of the ROI for static camera
                "render_mode": None,
            }
        )
        return cfg

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        # If render_mode is in config but not passed as arg, extract it
        if render_mode is None and config and "render_mode" in config:
            render_mode = config["render_mode"]

        # Initialize ID tracking BEFORE super().__init__ because it calls reset()
        self.next_vehicle_id = 100

        super().__init__(config, render_mode)

        # Manually set a spec to avoid assertions in gym/highway-env rendering
        from gymnasium.envs.registration import EnvSpec

        self.spec = EnvSpec(id="MergeInteraction-v0")

        # Apply visual patches for ID rendering
        self._apply_visual_patches()

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Build a road network with a highway and a merging ramp with an acceleration lane.
        """
        net = RoadNetwork()

        # Configuration parameters
        h_lanes = self.config["highway_lanes"]
        r_lanes = self.config["ramp_lanes"]

        # Section lengths
        l_pre = self.config["ramp_distance"]
        l_merge = self.config["merging_length"]
        l_post = self.config["post_merge_length"]
        l_curve = self.config["ramp_curve_length"]

        w = self.config["lane_width"]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        # --- Highway ---
        for i in range(h_lanes):
            lt_left = c if i == 0 else n
            lt_right = c if i == h_lanes - 1 else s
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, i * w], [l_pre, i * w], line_types=[lt_left, lt_right]
                ),
            )
            lt_right_merge = s if i == h_lanes - 1 else lt_right
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [l_pre, i * w],
                    [l_pre + l_merge, i * w],
                    line_types=[lt_left, lt_right_merge],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [l_pre + l_merge, i * w],
                    [l_pre + l_merge + l_post, i * w],
                    line_types=[lt_left, lt_right],
                ),
            )

        # --- Ramp ---
        ramp_y_start = h_lanes * w + self.config["ramp_offset"]
        for i in range(r_lanes):
            net.add_lane(
                "j",
                "k",
                StraightLane(
                    [0, ramp_y_start + i * w],
                    [l_pre - l_curve, ramp_y_start + i * w],
                    line_types=[c, c],
                ),
            )
            y_accel = (h_lanes + i) * w
            y_ramp = ramp_y_start + i * w
            y_mid = (y_ramp + y_accel) / 2
            amplitude = (y_ramp - y_accel) / 2
            net.add_lane(
                "k",
                "b",
                SineLane(
                    [l_pre - l_curve, y_mid],
                    [l_pre, y_mid],
                    amplitude,
                    2 * np.pi / (2 * l_curve),
                    np.pi / 2,
                    line_types=[c, c],
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [l_pre, y_accel], [l_pre + l_merge, y_accel], line_types=[s, c]
                ),
            )

        self.road = Road(network=net, np_random=self.np_random, record_history=False)

    def _make_vehicles(self) -> None:
        """Initial population of vehicles maintaining count and ratio."""
        target_count = self.config["target_vehicles_count"]
        cav_ratio = self.config["cav_ratio"]
        lanes = self.road.network.all_side_lanes(
            ("a", "b", 0)
        ) + self.road.network.all_side_lanes(("j", "k", 0))
        spawn_step = (self.config["ramp_distance"] + self.config["merging_length"]) / (
            target_count / len(lanes)
        )

        count = 0
        for lane_index in lanes:
            lane = self.road.network.get_lane(lane_index)
            for d in np.arange(0, lane.length, spawn_step):
                if count >= target_count:
                    break
                is_cav = self.np_random.uniform() < cav_ratio
                if lane_index[0] == "j":  # Ramp
                    route = [
                        ("j", "k", lane_index[2]),
                        ("k", "b", lane_index[2]),
                        ("b", "c", self.config["highway_lanes"] - 1),
                        ("c", "d", self.config["highway_lanes"] - 1),
                    ]
                else:  # Highway
                    route = [
                        ("a", "b", lane_index[2]),
                        ("b", "c", lane_index[2]),
                        ("c", "d", lane_index[2]),
                    ]
                self._spawn_vehicle(lane_index, d, is_cav, route)
                count += 1

        # --- Add "Small Circle" Obstacle at Ramp Tip (Paper Scenario) ---
        l_pre = self.config["ramp_distance"]
        l_merge = self.config["merging_length"]
        h_lanes = self.config["highway_lanes"]
        w = self.config["lane_width"]
        # Position at the very end of the merging lane
        obstacle_pos = [l_pre + l_merge, h_lanes * w]
        obs = Obstacle(self.road, obstacle_pos)
        obs.LENGTH = 1.2
        obs.WIDTH = 1.2
        obs.id = 999
        obs.color = (255, 0, 0)  # Red
        self.road.objects.append(obs)

        if self.road.vehicles:
            self.vehicle = self.road.vehicles[0]

    def _spawn_vehicle(
        self,
        lane_index: Tuple,
        position_on_lane: float,
        is_cav: bool,
        route: Optional[List] = None,
    ) -> Vehicle:
        lane = self.road.network.get_lane(lane_index)
        speed = self.np_random.uniform(*self.config["initial_speed_range"])

        # Instantiate using specific classes based on vehicle type
        pos = lane.position(position_on_lane, 0)
        if is_cav:
            vehicle = CAVVehicle(
                self.road, pos, heading=lane.heading_at(position_on_lane), speed=speed
            )
        else:
            vehicle = IDMVehicle(
                self.road, pos, heading=lane.heading_at(position_on_lane), speed=speed
            )

        vehicle.id = self.next_vehicle_id
        self.next_vehicle_id += 1
        vehicle.LENGTH = self.config["vehicle_length"]
        vehicle.WIDTH = self.config["vehicle_width"]
        vehicle.is_cav = is_cav
        if route:
            vehicle.route = route

        if not is_cav:
            self._randomize_behavior(vehicle)
            vehicle.color = (0, 0, 255)  # HDV (Blue)
        else:
            # Set a distinct color for CAVs (Yellow)
            vehicle.color = (255, 255, 0)
            vehicle.is_cav = True
            # We still need basic IDM parameters for fallback, but they don't move based on IDM
            vehicle.target_speed = self.np_random.uniform(
                *self.config["target_speed_range"]
            )

        self.road.vehicles.append(vehicle)
        return vehicle

    def _randomize_behavior(self, vehicle: IDMVehicle) -> None:
        """Assign independent behavioral profiles to each vehicle."""
        vehicle.TIME_WANTED = self.np_random.uniform(*self.config["time_wanted_range"])
        vehicle.POLITENESS = self.np_random.uniform(*self.config["politeness_range"])
        vehicle.DISTANCE_WANTED = self.np_random.uniform(
            *self.config["distance_wanted_range"]
        )
        vehicle.LANE_CHANGE_MIN_ACC_GAIN = self.np_random.uniform(
            *self.config["min_acc_gain_range"]
        )
        vehicle.ACC_MAX = self.np_random.uniform(*self.config["acc_max_range"])

        vehicle.target_speed = self.np_random.uniform(
            *self.config["target_speed_range"]
        )

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.latest_removal_events = []  # Clear at start of each step
        obs, reward, terminated, truncated, info = super().step(action)
        self._handle_infinite_flow()
        self._enforce_fixed_camera()

        # Reset the manual action flag AFTER the entire environment step is complete
        for v in self.road.vehicles:
            if getattr(v, "is_cav", False):
                v._manual_action_applied = False

        return obs, reward, terminated, truncated, info

    def _handle_infinite_flow(self) -> None:
        scene_limit = (
            self.config["ramp_distance"]
            + self.config["merging_length"]
            + self.config["post_merge_length"]
        )
        to_remove = []
        current_cav_count = 0
        for v in self.road.vehicles:
            if v.position[0] > scene_limit - 10:
                to_remove.append(v)
                self.latest_removal_events.append({
                    "id": v.id, "reason": "graduated", "x": v.position[0], "is_cav": getattr(v, "is_cav", False)
                })
            elif v.crashed:
                to_remove.append(v)
                self.latest_removal_events.append({
                    "id": v.id, "reason": "crashed", "x": v.position[0], "is_cav": getattr(v, "is_cav", False)
                })
            elif getattr(v, "is_cav", False):
                current_cav_count += 1

            # Application of visual markers (Safe here)
            if getattr(v, "is_cav", False):
                v.color = (255, 255, 0)  # Yellow
            else:
                v.color = (0, 0, 255)  # Blue

        for v in to_remove:
            self.road.vehicles.remove(v)

        target_count = self.config["target_vehicles_count"]
        target_cav_ratio = self.config["cav_ratio"]
        origin_lanes = [("a", "b", i) for i in range(self.config["highway_lanes"])] + [
            ("j", "k", i) for i in range(self.config["ramp_lanes"])
        ]

        while len(self.road.vehicles) < target_count:
            lane_idx = origin_lanes[self.np_random.integers(len(origin_lanes))]
            lane = self.road.network.get_lane(lane_idx)
            is_clear = True
            spawn_pos = lane.position(0, 0)
            for v in self.road.vehicles:
                # Calculate longitudinal distance
                long_dist = abs(v.position[0] - spawn_pos[0])

                if v.lane_index == lane_idx:
                    # Same lane: respect the full spawn gap
                    if long_dist < self.config["spawn_gap"]:
                        is_clear = False
                        break
                elif long_dist < 1.0:  # Lateral safety: 1m buffer in adjacent lanes
                    is_clear = False
                    break
            if is_clear:
                current_ratio = current_cav_count / max(1, len(self.road.vehicles))
                # Robust CAV check
                should_be_cav = (target_cav_ratio >= 1.0) or (
                    current_ratio < target_cav_ratio
                )
                if lane_idx[0] == "j":
                    route = [
                        ("j", "k", lane_idx[2]),
                        ("k", "b", lane_idx[2]),
                        ("b", "c", self.config["highway_lanes"] - 1),
                        ("c", "d", self.config["highway_lanes"] - 1),
                    ]
                else:
                    route = [
                        ("a", "b", lane_idx[2]),
                        ("b", "c", lane_idx[2]),
                        ("c", "d", lane_idx[2]),
                    ]
                v = self._spawn_vehicle(lane_idx, 0, should_be_cav, route)
                if should_be_cav:
                    current_cav_count += 1
            else:
                break

    def _enforce_fixed_camera(self) -> None:
        if self.config["fixed_camera"] and self.viewer:
            self.viewer.sim_surface.scaling = self.config["scaling"]
            roi_x = self.config["roi_center"]
            self.viewer.window_position = lambda: np.array([roi_x, 0])
            self.viewer.observer_vehicle = None

    def get_vehicles_info(self) -> List[Dict[str, Any]]:
        info = []
        for i, v in enumerate(self.road.vehicles):
            if v.crashed:
                continue

            is_moving = v.speed > 0.1

            info.append(
                {
                    "index": i,
                    "id": getattr(v, "id", i),
                    "position": v.position.tolist(),
                    "velocity": v.velocity.tolist(),
                    "speed": v.speed,
                    "heading": v.heading,
                    "lane_index": getattr(v, "lane_index", None),
                    "is_cav": getattr(v, "is_cav", False),
                    "is_moving": is_moving,
                    "length": v.LENGTH,
                    "width": v.WIDTH,
                    "crashed": v.crashed,
                }
            )
        for i, obj in enumerate(self.road.objects):
            info.append(
                {
                    "index": len(self.road.vehicles) + i,
                    "id": getattr(obj, "id", 999),
                    "position": obj.position.tolist(),
                    "velocity": [0.0, 0.0],
                    "heading": obj.heading,
                    "lane_index": getattr(obj, "lane_index", None),
                    "is_cav": False,
                    "length": obj.LENGTH,
                    "width": obj.WIDTH,
                    "crashed": False,
                }
            )
        return info

    def _reward(self, action: int) -> float:
        return 0.0

    def _is_terminated(self) -> bool:
        return False

    def _is_truncated(self) -> bool:
        return False

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._enforce_fixed_camera()
        frame = super().render()
        return frame

    def save_frame(self, filename: str):
        pass  # Optional matplotlib save logic here

    def _apply_visual_patches(self):
        from highway_env.vehicle.graphics import VehicleGraphics
        import pygame

        if not hasattr(VehicleGraphics, "_original_display"):
            VehicleGraphics._original_display = VehicleGraphics.display

        def patched_display(
            cls,
            vehicle,
            surface,
            transparent=False,
            offscreen=False,
            label=False,
            draw_roof=False,
        ):
            VehicleGraphics._original_display(
                vehicle,
                surface,
                transparent,
                offscreen,
                label=False,
                draw_roof=draw_roof,
            )
            if not offscreen:
                v_id = getattr(vehicle, "id", None)
                if v_id is not None:
                    try:
                        font = pygame.font.Font(None, 15)
                        text = font.render(f"#{v_id}", 1, (10, 10, 10), (255, 255, 255))
                        pos = surface.pos2pix(vehicle.position[0], vehicle.position[1])
                        surface.blit(text, (pos[0], pos[1] - 12))
                    except:
                        pass

        VehicleGraphics.display = classmethod(patched_display)
