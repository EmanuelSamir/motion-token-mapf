import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.planning.motion_primitives import MotionPrimitives
import highway_env.utils as utils

class PPOMergeWrapper(gym.Wrapper):
    """
    Adapts MergeInteractionEnv for single-agent PPO training following the BK-PBS paper.
    - Discrete Action Space (6 Primitives)
    - Paper-specific Reward Function
    - Handles single-agent control over the 'ego' vehicle (CAV)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Define mapping from discrete actions to primitives
        self.action_map = {
            0: "IDLE",
            1: "ACCELERATE",
            2: "DECELERATE",
            3: "HARD_BRAKE",
            4: "LANE_CHANGE_LEFT",
            5: "LANE_CHANGE_RIGHT"
        }
        
        # Motion Primitives helper (for low-level control mapping)
        self.primitives = MotionPrimitives(road=None) # Will update road in step
        
        # Override action space to discrete (6 primitives)
        self.action_space = gym.spaces.Discrete(6)
        
        # Constants from paper/configs
        self.v_max = self.env.unwrapped.config.get("target_speed_range", [30, 35])[1]
        self.v_min = self.env.unwrapped.config.get("target_speed_range", [30, 35])[0]
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Map discrete action to maneuver string
        maneuver = self.action_map.get(int(action), "IDLE")
        
        # 2. Get low-level control (acceleration, steering) from MotionPrimitives
        # We need the current ego state
        ego = self.env.unwrapped.vehicle
        self.primitives.road = self.env.unwrapped.road
        
        # State: [x, y, v, h, lane_idx]
        # We need to find the lane index geometrically or use stored one
        # Current state extraction matching MotionPrimitives expectation
        ego_state = np.array([
            ego.position[0],
            ego.position[1],
            ego.speed,
            ego.heading,
            getattr(ego, "lane_index", ("", "", 0))[2]
        ])
        
        low_level_action = self.primitives.get_low_level_action(ego_state, maneuver)
        
        # 3. Apply action DIRECTLY to the ego vehicle
        # We pass the dictionary to the vehicle, and then call env.step(None)
        # to bypass the simulator's action_type wrapper that was causing crashes.
        ego.act(low_level_action)
        
        # 4. Step the environment (Advancing physics only)
        obs, _, _, truncated, info = self.env.step(None) 
        
        # 4. Calculate Reward (Paper: Speed + Collision)
        reward = self._calculate_reward(ego)
        
        # 5. Handle Termination (Collision, Off-road, or Duration)
        # We terminate on crash or if the vehicle leaves the road
        terminated = ego.crashed or not ego.on_road
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, vehicle) -> float:
        """
        Implementation of the reward function from BK-PBS Paper:
        - "reward function of maintaining a target speed and avoiding collision"
        """
        # Collision or Off-road Penalty
        if vehicle.crashed or not vehicle.on_road:
            return -10.0
        
        # Speed Reward
        # Reward is higher as we approach v_max
        # normalized_speed = (v - v_min) / (v_max - v_min)
        # We use a simple linear progression
        speed_reward = vehicle.speed / self.v_max
        
        # Optional: Lane centering/alignment to stabilize training (common in highway-env)
        # But for benchmark fidelity, we keep it simple.
        
        return float(speed_reward)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(**kwargs)
