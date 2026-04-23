import hydra
import os
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.env.merge_interaction_env import MergeInteractionEnv
from src.env.ppo_wrapper import PPOMergeWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("--- Starting PPO Benchmark Training ---")
    
    # 1. Environment Preparation
    # Convert Hydra config to dictionary for highway-env compatibility
    env_params = OmegaConf.to_container(cfg.env, resolve=True)
    
    # Ensure correct action type for the wrapper (wrapper handles discrete-to-primitive)
    env_params["action"] = {
        "type": "ContinuousAction",
        "acceleration_range": [-6.0, 6.0],
        "steering_range": [-np.pi/3, np.pi/3],
        "clip": True
    }
    
    # Create the base environment
    base_env = MergeInteractionEnv(config=env_params)
    
    # Wrap for single-agent PPO logic and paper-specific rewards
    env = PPOMergeWrapper(base_env)
    
    # Monitor log recording (essential for rewards tracking)
    log_dir = "./logs/ppo_benchmark/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # 2. Model Configuration (According to BK-PBS Paper V-C)
    # - Two layers, 256 hidden dimensions
    # - LR: 0.001, Gamma: 0.95, Batch: 512, Epochs: 10
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[256, 256], qf=[256, 256]) if hasattr(torch.nn, "qf") else [256, 256] 
    )
    # Simple list for MLP policy architecture in SB3 PPO is: [256, 256]
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=0.001,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard/ppo_benchmark/"
    )

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./checkpoints/ppo_benchmark/",
        name_prefix="ppo_merge_model"
    )

    # 4. Training Loop
    total_timesteps = 5000000 # ~4-6 hours at current FPS
    print(f"Training for {total_timesteps} steps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # 5. Save Final Model
    model.save("checkpoints/ppo_merge_final")
    print("Training Complete. Model saved to checkpoints/ppo_merge_final.zip")

if __name__ == "__main__":
    main()
