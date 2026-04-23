import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import torch
import os

from collections import deque

from src.env.merge_interaction_env import MergeInteractionEnv, CAVVehicle
from src.planning.motion_primitives import MotionPrimitives
from src.planning.bk_planner import BKPlanner
from src.planning.tree_visualizer import TreeVisualizer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def test_bk_planner(cfg: DictConfig):
    # 1. Setup Environment
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_config["cav_ratio"] = cfg.get("test_cav_ratio", 0.5)
    env_config["target_vehicles_count"] = 20  # Increased for stress testing
    env_config["render_mode"] = cfg.env.get("render_mode", "human")

    print(
        f"\n[BK-TEST] Initializing Environment with {env_config['cav_ratio'] * 100:.0f}% CAV Ratio and {env_config['target_vehicles_count']} vehicles..."
    )
    env = MergeInteractionEnv(env_config)
    obs, info = env.reset()

    # 2. Setup BK-Planner
    mp = MotionPrimitives(road=env.unwrapped.road)

    # Path to the model being trained (change this to your actual checkpoint)
    ckpt_path = cfg.get("ckpt_path", "checkpoints/latest.ckpt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(ckpt_path):
        print(
            f"⚠️ [BK-TEST] Checkpoint not found at {ckpt_path}. Planner will likely fail to load."
        )
        print(
            f"   Please provide a valid path via 'ckpt_path=...' or ensure 'checkpoints/latest.ckpt' exists."
        )

    planner = BKPlanner(
        mp,
        model_path=ckpt_path,
        device=device,
        dt=0.2,
        safe_radius=env_config.get("collision_radius", 7.0),
        margin_l=env_config.get("safety_margin_l", 0.20),
        margin_w=env_config.get("safety_margin_w", 0.20),
        horizon_agg=200.0,  # Increased D
        horizon_safe=120.0,  # Increased D'
    )

    # 3. Identify Agents & Initial Warm-up (10 steps)
    init_agents = [v for v in env.unwrapped.road.vehicles]
    
    # Limitation due to MotionLM `max_agents` constraint
    MAX_AGENTS = 10
    if len(init_agents) > MAX_AGENTS:
        init_agents = sorted(init_agents, key=lambda v: v.position[0], reverse=True)[:MAX_AGENTS]

    agent_types = {v.id: ("CAV" if isinstance(v, CAVVehicle) else "HDV") for v in init_agents}
    planner.set_agent_types(agent_types)

    agents_history = {v.id: deque(maxlen=10) for v in init_agents}
    
    print(f"[BK-TEST] Warming up for 10 steps (Constant Velocity for CAVs)...")
    for ws in range(10):
        for v in init_agents:
            # Record state: [x, y, vx, vy, heading]
            vx = v.speed * np.cos(v.heading)
            vy = v.speed * np.sin(v.heading)
            agents_history[v.id].append([v.position[0], v.position[1], vx, vy, v.heading])
            
            # Action: Constant Velocity for CAVs
            if agent_types[v.id] == "CAV":
                v.act({"acceleration": 0.0, "steering": 0.0})
            else:
                v.act(None) # HDVs follow IDM/MOBIL
        
        env.step(None)
        if env_config.get("render_mode") == "human":
            env.render()
            time.sleep(0.01)

    # 4. Prepare planning states
    agents_states = {}
    active_history = {}

    for v in init_agents:
        # [x, y, v, h, lane_idx]
        state = np.array([
            v.position[0],
            v.position[1],
            v.speed,
            v.heading,
            env.unwrapped.road.network.get_closest_lane_index(v.position, v.heading)[2]
        ])
        agents_states[v.id] = state
        active_history[v.id] = list(agents_history[v.id])

    print(f"[BK-TEST] Planning for {len(init_agents)} agents. Types: {agent_types}")

    # 5. Perform One-Shot BK-PBS Planning
    print("[BK-TEST] Running BK-PBS High-Level Search...")
    v_info = env.unwrapped.get_vehicles_info()

    try:
        results = planner.plan_all(agents_states, v_info=v_info, agents_history=active_history)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ [BK-TEST] Planning failed with error: {e}")
        env.close()
        return

    # 6. Execution Loop (Visualize the plan)
    print(f"\n[BK-TEST] Starting Execution. New arrivals will be FROZEN.")

    # We find the min path length to determine the "duration" of this specific plan
    valid_paths = [
        len(res.get("path", [])) for res in results.values() if res.get("path")
    ]
    duration = min(valid_paths) if valid_paths else 50

    print(f"[BK-TEST] Duration: {duration}")
    print("[BK-TEST] Planes Generados por el BK-PBS:")
    for v_id, res in results.items():
        v_type = agent_types.get(v_id, "Desconocido")
        actos = res.get("actions", [])
        print(f"  -> ID: {v_id} | Tipo: {v_type} | Cantidad de acciones: {len(actos)}")
        if actos:
            print(f"       Acciones: {actos}")

    plot_planned_trajectories(
        results, agent_types, output_filename="bk_planned_trajectories.gif"
    )

    for step in range(duration):
        # Apply planned actions
        for v in init_agents:
            if v.id in results:
                res = results[v.id]
                if res.get("phase") == "PREDICTED":
                    # HDV Ghost execution for verification
                    path = res.get("path", [])
                    if step < len(path):
                        v.position = path[step][:2]
                        v.heading = path[step][3]
                        v.speed = path[step][2]
                else:
                    # CAV Primitive execution
                    actions = res.get("actions", [])
                    if step < len(actions):
                        maneuver = actions[step]
                        curr_state = np.array(
                            [
                                v.position[0],
                                v.position[1],
                                v.speed,
                                v.heading,
                                env.unwrapped.road.network.get_closest_lane_index(
                                    v.position, v.heading
                                )[2],
                            ]
                        )
                        low_level = mp.get_low_level_action(curr_state, maneuver)
                        v.act(low_level)

        # Freeze non-initial agents
        for v in env.unwrapped.road.vehicles:
            if v.id not in agent_types:
                v.speed = 0
                v.act({"acceleration": 0.0, "steering": 0.0})

        env.step(None)
        env.render()
        time.sleep(0.02)  # Slightly faster playback

        if step % 20 == 0:
            print(f"Step {step}/{duration}")

    print("\n[BK-TEST] Plan execution finished. Pausing for inspection...")
    time.sleep(5)
    print("[BK-TEST] Completed.")
    env.close()


def plot_planned_trajectories(
    results, agent_types, output_filename="bk_planned_trajectories.gif"
):
    """
    Genera un GIF animado de las trayectorias planificadas por el BK-PBS,
    usando matplotlib con un legend que indica el tipo de vehiculo (HDV o CAV).
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(10, 6))

    max_steps = 0
    all_x, all_y = [], []
    for res in results.values():
        path = res.get("path", [])
        if len(path) > max_steps:
            max_steps = len(path)
        for state in path:
            all_x.append(state[0])
            all_y.append(state[1])

    if max_steps == 0 or not all_x:
        print("[Plot] No hay trayectorias para graficar.")
        return

    min_x, max_x = min(all_x) - 10, max(all_x) + 10
    min_y, max_y = min(all_y) - 5, max(all_y) + 5

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_title("Trayectorias Planificadas (BK-PBS)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    lines = {}
    scatters = {}
    texts = {}

    import matplotlib.cm as cm

    try:
        cmap = plt.colormaps["tab20"]
    except AttributeError:
        cmap = cm.get_cmap("tab20")

    for i, (v_id, res) in enumerate(results.items()):
        v_type = agent_types.get(v_id, "Desconocido")
        c = cmap(i % 20)

        # CAV normal line, HDV dotted line with less opacity
        ls = "-" if v_type == "CAV" else "--"
        alpha = 1.0 if v_type == "CAV" else 0.5

        (lines[v_id],) = ax.plot(
            [],
            [],
            color=c,
            label=f"ID: {v_id} ({v_type})",
            linestyle=ls,
            alpha=alpha,
            linewidth=2,
        )
        scatters[v_id] = ax.scatter([], [], color=c, s=60, zorder=5)
        texts[v_id] = ax.text(
            0, 0, str(v_id), fontsize=8, color="black", fontweight="bold"
        )

    # Legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    plt.tight_layout()

    def update(frame):
        artists = []
        for v_id, res in results.items():
            path = res.get("path", [])
            idx = min(frame, len(path) - 1)
            if idx >= 0:
                state = path[idx]
                x, y = state[0], state[1]

                scatters[v_id].set_offsets([[x, y]])
                texts[v_id].set_position((x, y + 1.5))

                history_x = [s[0] for s in path[: idx + 1]]
                history_y = [s[1] for s in path[: idx + 1]]
                lines[v_id].set_data(history_x, history_y)

                artists.extend([scatters[v_id], texts[v_id], lines[v_id]])
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=max_steps, interval=200, blit=True
    )

    print(f"\n[Plot] Procesando animación y guardando en '{output_filename}'...")
    try:
        anim.save(output_filename, writer="pillow")
        print(f"[Plot] Guardado exitoso.\n")
    except Exception as e:
        print(f"[Plot] Error al guardar GIF: {e}\n")
    plt.close()


if __name__ == "__main__":
    test_bk_planner()
