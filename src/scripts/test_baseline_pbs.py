import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import os

from src.env.merge_interaction_env import MergeInteractionEnv, CAVVehicle
from src.planning.motion_primitives import MotionPrimitives
from src.planning.pbs_planner import PBSPlanner


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def test_baseline_pbs(cfg: DictConfig):
    # 1. Setup Environment
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_config["cav_ratio"] = cfg.get("test_cav_ratio", 0.5)
    env_config["target_vehicles_count"] = 20  # Same as BK-PBS test
    env_config["render_mode"] = "human"

    print(
        f"\n[BASELINE-PBS] Initializing Environment with {env_config['cav_ratio'] * 100:.0f}% CAV Ratio and {env_config['target_vehicles_count']} vehicles..."
    )
    env = MergeInteractionEnv(env_config)
    obs, info = env.reset()

    # 2. Setup Baseline PBS Planner
    mp = MotionPrimitives(road=env.unwrapped.road)
    planner = PBSPlanner(
        mp,
        dt=0.2,
        safe_radius=env_config.get("collision_radius", 7.0),
        margin_l=env_config.get("safety_margin_l", 0.20),
        margin_w=env_config.get("safety_margin_w", 0.20),
        horizon_agg=250.0,
        horizon_safe=130.0,
    )

    # 3. Identify Agents & Types
    init_agents = [v for v in env.unwrapped.road.vehicles]
    agent_types = {}
    agents_states = {}

    for v in init_agents:
        v_type = "CAV" if isinstance(v, CAVVehicle) else "HDV"
        agent_types[v.id] = v_type

        # [x, y, v, h, lane_idx]
        state = np.array(
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
        agents_states[v.id] = state

    # Crucial: Specify which agents the baseline planner can "move" vs "treat as static"
    planner.set_agent_types(agent_types)
    print(
        f"[BASELINE-PBS] Planning for {len(init_agents)} agents. Types: {agent_types}"
    )
    print(f"               (In this baseline, HDVs are treated as IMMOBILE obstacles)")

    # 4. Perform One-Shot Planning
    print("[BASELINE-PBS] Running Baseline PBS Search...")
    v_info = env.unwrapped.get_vehicles_info()
    results = planner.plan_all(agents_states, v_info=v_info)

    # 5. Execution Loop
    print(f"\n[BASELINE-PBS] Starting Execution. New arrivals will be FROZEN.")

    # Calculate duration based on the shortest planned path
    valid_paths = [
        len(res.get("path", [])) for res in results.values() if res.get("path")
    ]
    duration = min(valid_paths) if valid_paths else 50

    print(f"[BASELINE-PBS] Duration: {duration}")
    print("[BASELINE-PBS] Planes Generados por el PBS:")
    for v_id, res in results.items():
        v_type = agent_types.get(v_id, "Desconocido")
        actos = res.get("actions", [])
        print(f"  -> ID: {v_id} | Tipo: {v_type} | Cantidad de acciones: {len(actos)}")
        if actos:
            print(f"       Acciones: {actos}")

    plot_planned_trajectories(
        results, agent_types, output_filename="planned_trajectories.gif"
    )

    for step in range(duration):
        for v in init_agents:
            if v.id in results:
                res = results[v.id]
                v_type = agent_types.get(v.id, "HDV")
                if v_type == "CAV":
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
                else:
                    # HDVs in baseline are ignored by the planner logic,
                    # they just follow their default simulator IDM behavior.
                    pass

        # Step Simulation
        env.step(None)
        env.render()
        time.sleep(0.2)  # Faster playback

        if step % 20 == 0:
            print(f"Step {step}/{duration}")

    print("\n[BASELINE-PBS] Plan execution finished. Pausing for inspection...")
    time.sleep(5)
    print("[BASELINE-PBS] Completed.")
    env.close()


def plot_planned_trajectories(
    results, agent_types, output_filename="planned_trajectories.gif"
):
    """
    Genera un GIF animado de las trayectorias planificadas por el PBS,
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
    ax.set_title("Trayectorias Planificadas (PBS Baseline)")
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
    test_baseline_pbs()
