import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import os
from datetime import datetime
from tqdm import tqdm

from src.env.merge_interaction_env import MergeInteractionEnv
from src.env.wrappers import TrajectoryCollectorWrapper

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Access parameters directly from Config
    num_episodes = cfg.get("num_episodes", 10)
    render_mode = "human" if cfg.render else None
    seed = cfg.get("seed", 42)
    
    # 1. Setup Environment Configuration
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    
    # Force Pure HDV Configuration for model training
    env_config["cav_ratio"] = 0.0
    env_config["render_mode"] = render_mode
    
    duration = env_config.get("duration", 200)
    
    print("\n" + "🚀" + "="*50)
    print(" 🚗 HDV TRAJECTORY DATA COLLECTOR")
    print("="*52)
    print(f" 📂 Output Dir:  data/trajectories/")
    print(f" 🎬 Episodes:    {num_episodes}")
    print(f" ⏱️ Duration:    {duration} steps")
    print(f" 🤖 CAV Ratio:   0.0 (Pure HDV)")
    print(f" 👁️ Render:      {render_mode}")
    print(f" 🌱 Seed:        {seed}")
    print("="*52 + "\n")
    
    # 2. Initialize Environment
    raw_env = MergeInteractionEnv(env_config, render_mode=render_mode)
    
    # 3. Wrap with Trajectory Collector
    # We use the 'tag' from config to name the subdirectory
    dataset_tag = cfg.get("tag", "hdv_raw_collection")
    output_dir = os.path.join("data/trajectories", dataset_tag)
    
    env = TrajectoryCollectorWrapper(
        raw_env, 
        output_dir=output_dir, 
        save_format="pkl"
    )
    
    # Set global seed
    np.random.seed(seed)
    
    # Check if we want to randomize vehicle density
    # Expected format: [min, max]
    vehicles_range = cfg.get("vehicles_range", None)
    
    try:
        total_steps = 0
        for ep in range(num_episodes):
            # Dynamic Randomization of Traffic Density
            if vehicles_range:
                target_count = np.random.randint(vehicles_range[0], vehicles_range[1] + 1)
                env.unwrapped.config["target_vehicles_count"] = target_count
                # env.unwrapped.configure({"target_vehicles_count": target_count}) # Alternative
            
            obs, info = env.reset(seed=seed + ep)
            
            pbar = tqdm(total=duration, desc=f" Episode {ep+1}/{num_episodes} (Veh: {env.unwrapped.config['target_vehicles_count']})", unit="step")
            
            ep_step = 0
            while ep_step < duration:
                # HDVs are controlled internally by IDM/MOBIL
                obs, reward, terminated, truncated, info = env.step(None)
                
                if render_mode == "human":
                    env.render()
                    time.sleep(env_config.get("step_delay", 0.01))
                
                ep_step += 1
                total_steps += 1
                pbar.update(1)
                
                if terminated or truncated:
                    break

            pbar.close()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Collection interrupted by user.")
    finally:
        print(f"\n📊 Collection Finished. Total steps recorded: {total_steps}")
        env.close()

if __name__ == "__main__":
    main()
