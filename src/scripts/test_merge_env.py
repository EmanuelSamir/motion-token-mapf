import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from src.env.merge_interaction_env import MergeInteractionEnv
from src.env.wrappers import TrajectoryCollectorWrapper


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def test_env(cfg: DictConfig):
    # 1. Load config from Hydra
    # We resolve the DictConfig to a plain dict for the environment
    env_config = OmegaConf.to_container(cfg.env, resolve=True)

    print("Environment Initializing with Hydra Config:")
    print(OmegaConf.to_yaml(cfg.env))

    # 2. Instantiate Env
    env = MergeInteractionEnv(env_config)

    # 3. Apply Data Collection Wrapper if enabled
    collect_cfg = cfg.env.get("collection", {})
    if collect_cfg.get("enabled", False):
        env = TrajectoryCollectorWrapper(
            env,
            output_dir=collect_cfg.get("output_dir", "data/trajectories"),
            save_format=collect_cfg.get("save_format", "pkl"),
        )
        print(f"Data Collection ENABLED. Saving to {collect_cfg.get('output_dir')}")

    num_episodes = (
        collect_cfg.get("num_episodes", 1) if collect_cfg.get("enabled", False) else 1
    )

    control_cavs = False  # cfg.env.get("control_cavs", False)
    print(f"Modo control_cavs (CAVs controlados con veloc. constante): {control_cavs}")

    for ep in range(num_episodes):
        print(f"\n--- Starting Episode {ep} ---")
        obs, info = env.reset()

        # 4. Simulation Loop
        duration = env_config.get("duration", 200)
        for step in range(duration):
            if control_cavs:
                for v in env.unwrapped.road.vehicles:
                    if getattr(v, "is_cav", False):
                        # Control manual: velocidad constante (0 aceleración, 0 giro)
                        v.act({"acceleration": 0.0, "steering": 0.0})

            obs, reward, terminated, truncated, info = env.step(None)

            # Using unwrapped to access the info method reliably
            vehicles_info = env.unwrapped.get_vehicles_info()
            cav_count = sum(1 for v in vehicles_info if v["is_cav"])
            total_count = len(vehicles_info)

            if step % 100 == 0:
                print(
                    f"Step {step}: Total Vehicles={total_count}, CAVs={cav_count}, Ratio={cav_count / max(1, total_count):.2f}"
                )

            if env_config.get("render_mode") == "human":
                env.render()

    print("\nTest Completed.")
    env.close()


if __name__ == "__main__":
    test_env()
