from src.env.merge_interaction_env import MergeInteractionEnv, CAVVehicle
import numpy as np

def diag_rollout():
    config = MergeInteractionEnv.default_config()
    config["render_mode"] = None
    config["duration"] = 100
    config["cav_ratio"] = 1.0
    config["target_vehicles_count"] = 5
    
    env = MergeInteractionEnv(config)
    env.reset(seed=42)
    
    print(f"Goal X: 400.0")
    
    for step in range(100):
        # We don't call act() at all, let's see if they move naturally
        obs, reward, terminated, truncated, info = env.step(None)
        
        v_positions = [v.position[0] for v in env.road.vehicles]
        v_ids = [v.id for v in env.road.vehicles]
        print(f"Step {step}: IDs={v_ids}, PosX={[f'{p:.1f}' for p in v_positions]}")
        
    env.close()

if __name__ == "__main__":
    diag_rollout()
