import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import time
from typing import Dict, List, Any

from src.env.merge_interaction_env import MergeInteractionEnv
from src.planning.motion_primitives import MotionPrimitives
from src.planning.pbs_planner import PBSPlanner
import os

class ExecutionManager:
    """Manages active plans and execution pointers for all agents."""
    def __init__(self):
        self.plans = {} # v_id -> {"actions": [], "ptr": 0, "path": []}

    def add_plans(self, new_plans: Dict[int, Dict]):
        for v_id, data in new_plans.items():
            if data.get("path"):
                self.plans[v_id] = {
                    "actions": data["actions"],
                    "ptr": 0,
                    "path": data["path"]
                }

    def get_command(self, v_id) -> str:
        plan = self.plans.get(v_id)
        if plan and plan["ptr"] < len(plan["actions"]):
            return plan["actions"][plan["ptr"]]
        return "IDLE"

    def step_agent(self, v_id):
        if v_id in self.plans:
            self.plans[v_id]["ptr"] += 1

    def is_path_exhausted(self, v_id, threshold=5) -> bool:
        plan = self.plans.get(v_id)
        if not plan: return True
        return (len(plan["actions"]) - plan["ptr"]) <= threshold

    def cleanup(self, active_ids):
        """Removes plans for agents no longer in the simulation."""
        self.plans = {vid: p for vid, p in self.plans.items() if vid in active_ids}

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup Environment
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_config["render_mode"] = "human"
    env_config["cav_ratio"] = 1.0 # Force 100% Centralized CAVs
    
    print("\n" + "="*50)
    print("🚀 BK-PBS INFINITE FLOW RUNNER")
    print("="*50)
    
    env = MergeInteractionEnv(env_config)
    obs, info = env.reset()
    
    mp = MotionPrimitives(road=env.unwrapped.road)
    planner = PBSPlanner(
        mp, 
        dt=0.2, 
        safe_radius=env_config.get("collision_radius", 7.0),
        margin_l=env_config.get("safety_margin_l", 0.20),
        margin_w=env_config.get("safety_margin_w", 0.20)
    )
    manager = ExecutionManager()
    
    step = 0
    max_steps = 2000 # Simulation limit
    
    try:
        while step < max_steps:
            # A. Get Active Agents (Only CAVs)
            v_info = env.unwrapped.get_vehicles_info()
            active_ids = [v["id"] for v in v_info if v.get("is_cav", False) and not v["crashed"]]
            manager.cleanup(active_ids)
            
            # B. Trigger Re-planning if needed
            unplanned = [vid for vid in active_ids if vid not in manager.plans]
            exhausted = [vid for vid in active_ids if manager.is_path_exhausted(vid)]
            
            if unplanned or exhausted:
                reason = "New Agents" if unplanned else "Path Exhaustion"
                print(f"\n[Step {step}] 🔄 Re-planning ({reason}). Agents: {len(active_ids)}")
                
                # Capture current states for planner
                agents_states = {}
                for v in v_info:
                    if v["id"] in active_ids:
                        l_idx = v.get("lane_index", [None, None, 0])[2]
                        # Use plan's target lane if accessible to maintain continuity
                        if v["id"] in manager.plans:
                            p = manager.plans[v["id"]]
                            if p["ptr"] < len(p["path"]):
                                l_idx = p["path"][p["ptr"]][4]

                        agents_states[v["id"]] = np.array([
                            v["position"][0], v["position"][1], 
                            v["speed"], v["heading"], l_idx
                        ])
                
                # Global PBS Coordination
                new_plans = planner.plan_all(agents_states, v_info=v_info)
                manager.add_plans(new_plans)

            # C. Actuate Agents
            id_to_v = {v.id: v for v in env.unwrapped.road.vehicles}
            for vid in active_ids:
                v = id_to_v.get(vid)
                if v:
                    maneuver = manager.get_command(vid)
                    # Use current state for low-level bridge
                    l_idx = env.unwrapped.road.network.get_closest_lane_index(v.position, v.heading)[2]
                    state = np.array([v.position[0], v.position[1], v.speed, v.heading, l_idx])
                    
                    action = mp.get_low_level_action(state, maneuver)
                    v.act(action)
                    manager.step_agent(vid)

            # D. Step Simulation
            env.step(None)
            env.render()
            
            if step % 50 == 0:
                print(f"Step {step}/{max_steps} | Active Agents: {len(active_ids)}")
            
            step += 1
            time.sleep(env_config.get("step_delay", 0.02))

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
