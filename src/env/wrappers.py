import gymnasium as gym
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

class TrajectoryCollectorWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper that automatically collects and saves vehicle trajectories.
    Refactored for Agent-Centric storage (ID -> List[States]) to ensure
    persistence and causality for MotionLM training.
    """

    def __init__(
        self, 
        env: gym.Env, 
        output_dir: str = "data/trajectories", 
        save_format: str = "pkl",
        enabled: bool = True
    ):
        super().__init__(env)
        self.output_dir = output_dir
        self.save_format = save_format
        self.enabled = enabled
        
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Agent-Centric Storage: aid -> List of state dicts
        self.episode_trajectories = {} 
        self.episode_count = 0
        self.roadgraph = self._extract_roadgraph()
        self.current_step = 0

    def reset(self, **kwargs):
        # Save previous episode if data exists
        if self.episode_trajectories:
            self.save_episode()
            
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode-specific tracking
        self.episode_trajectories = {}
        self.current_step = 0
        
        # Record initial state
        self._record_step()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        if self.enabled:
            self._record_step()
            
            # Auto-save based on duration to prevent memory bloat in infinite envs
            env_duration = self.env.unwrapped.config.get("duration", 200)
            if self.current_step >= env_duration:
                self.save_episode()
                self.episode_trajectories = {}
                self.current_step = 0
                self.episode_count += 1
                
        return obs, reward, terminated, truncated, info

    def _record_step(self):
        """
        Captures the current state of all vehicles.
        Crucially, it appends to existing trajectories to ensure 
        'Persistence of Life' even if vehicles are removed from the active road.
        """
        if not hasattr(self.env.unwrapped, "get_vehicles_info"):
            return

        vehicles_info = self.env.unwrapped.get_vehicles_info()
        
        for v in vehicles_info:
            aid = v["id"]
            
            # Initialize trajectory if new agent
            if aid not in self.episode_trajectories:
                self.episode_trajectories[aid] = []
            
            # Store state with timestamp
            v_state = v.copy()
            v_state["step"] = self.current_step
            self.episode_trajectories[aid].append(v_state)

    def _extract_roadgraph(self, points_per_lane: int = 20) -> List[Dict[str, Any]]:
        """
        Extracts the road geometry (lanes) into a structured format.
        """
        roadgraph = []
        road = getattr(self.env.unwrapped, "road", None)
        if not road or not hasattr(road, "network"):
            return []
            
        for _from, to_dict in road.network.graph.items():
            for _to, lanes in to_dict.items():
                for lane_idx, lane in enumerate(lanes):
                    lane_points = []
                    # Sample points along the lane
                    s_vals = np.linspace(0, lane.length, points_per_lane)
                    for s in s_vals:
                        pos = lane.position(s, 0)
                        heading = lane.heading_at(s)
                        lane_points.append({
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "heading": float(heading)
                        })
                    
                    roadgraph.append({
                        "id": f"{_from}_{_to}_{lane_idx}",
                        "points": lane_points,
                        "length": float(lane.length)
                    })
        return roadgraph

    def save_episode(self):
        """Saves the Agent-Centric trajectories to disk."""
        if not self.episode_trajectories:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hdv_data_{timestamp}_ep{self.episode_count:03d}"
        
        save_dict = {
            "roadgraph": self.roadgraph,
            "agents": self.episode_trajectories, # Map: aid -> List[States]
            "metadata": {
                "env_config": self.env.unwrapped.config,
                "episode": self.episode_count,
                "timestamp": timestamp,
                "total_steps": self.current_step,
                "num_agents": len(self.episode_trajectories)
            }
        }

        if self.save_format == "pkl":
            path = os.path.join(self.output_dir, f"{filename}.pkl")
            with open(path, "wb") as f:
                pickle.dump(save_dict, f)
            print(f"DEBUG: Saved Agent-Centric Episode {self.episode_count} ({len(self.episode_trajectories)} agents) to {path}")
            
        elif self.save_format == "csv":
            # Flatten only for CSV export compatibility
            rows = []
            for aid, trajectory in self.episode_trajectories.items():
                rows.extend(trajectory)
            
            path = os.path.join(self.output_dir, f"{filename}.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"DEBUG: Saved CSV Episode {self.episode_count} to {path}")

    def close(self):
        if self.episode_trajectories:
            self.save_episode()
        super().close()
