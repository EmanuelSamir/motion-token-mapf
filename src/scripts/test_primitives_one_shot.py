import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import time

from src.env.merge_interaction_env import MergeInteractionEnv
from src.planning.motion_primitives import MotionPrimitives
from src.planning.prioritized_planner import PrioritizedPlanner
from src.planning.pbs_planner import PBSPlanner
from src.planning.tree_visualizer import TreeVisualizer
import os


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def test_primitives_one_shot(cfg: DictConfig):
    # 1. Force 100% CAV ratio for the test
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    env_config["cav_ratio"] = 1.0
    env_config["render_mode"] = "human"
    debug_mode = env_config.get("debug_mode", False)

    print("\n[TEST] Initializing Environment with 100% CAV Ratio...")
    env = MergeInteractionEnv(env_config)
    obs, info = env.reset()

    # 2. Setup Planner
    mp = MotionPrimitives(road=env.unwrapped.road)
    planner = PBSPlanner(
        mp,
        dt=0.2,
        safe_radius=env_config.get("collision_radius", 7.0),
        margin_l=env_config.get("safety_margin_l", 0.20),
        margin_w=env_config.get("safety_margin_w", 0.20),
        horizon_agg=env_config.get("planning_horizon_aggressive", 150.0),
        horizon_safe=env_config.get("planning_horizon_safety", 80.0),
    )

    # 3. Identify Initial Agents
    init_agents = [v for v in env.unwrapped.road.vehicles]
    init_ids = [v.id for v in init_agents]
    print(f"[TEST] Planning for {len(init_agents)} initial agents: {init_ids}")

    # 4. Perform One-Shot Multi-Agent Planning
    agents_states = {}
    for v in init_agents:
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

    v_info = env.unwrapped.get_vehicles_info()

    results = planner.plan_all(agents_states, v_info=v_info)

    # 4. Trajectory Data for Final Logging
    trajectory_real = []  # List of [step, vid, x, y, h, s]
    trajectory_planned = []  # List of [step, vid, x, y, h, s]

    # Audit environment spawning
    l_pre = env.unwrapped.config["ramp_distance"]
    l_merge = env.unwrapped.config["merging_length"]
    lanes_count = len(
        env.unwrapped.road.network.all_side_lanes(("a", "b", 0))
        + env.unwrapped.road.network.all_side_lanes(("j", "k", 0))
    )
    spawn_step = (l_pre + l_merge) / (
        env.unwrapped.config["target_vehicles_count"] / lanes_count
    )
    spawn_gap = env.unwrapped.config.get("spawn_gap", 0)
    print(
        f"\n[ENV AUDIT] Target Count: {env.unwrapped.config['target_vehicles_count']} | Lanes: {lanes_count}"
    )
    print(
        f"            Spawn Step: {spawn_step:.2f}m | Spawn Gap Requirement: {spawn_gap:.2f}m"
    )
    if spawn_step < spawn_gap:
        print(
            "            ⚠️ [WARNING] OVERSPAWNING: spawn_step is smaller than spawn_gap. Overlaps likely at T=0."
        )
    elif spawn_step < 10.0:
        print(
            "            ⚠️ [WARNING] DENSITY: spawn_step is very small (<10m). Tight coordination required."
        )

    # Check if all planned successfully

    # Check if all planned successfully
    failed_ids = [v_id for v_id, res in results.items() if res.get("phase") == "FAILED"]
    if failed_ids:
        print(f"WARNING: Planning failed for agents: {failed_ids}")

    # 5. VISUALIZATION: Save Search Trees (Optional)
    if debug_mode:
        print("\n[TEST] Generating Search Tree Visualizations...")
        os.makedirs("debug_plots/trees", exist_ok=True)
        for v_id, res in results.items():
            tree_log = res.get("tree_log", [])
            chosen_ids = res.get("chosen_ids", [])
            phase = res.get("phase", "AGGRESSIVE")
            reason = res.get("reason", "SUCCESS")
            tree_path = f"debug_plots/trees/agent_{v_id}.png"
            TreeVisualizer.draw_tree(
                tree_log, chosen_ids, tree_path, v_id, phase, reason
            )

    # Reality Check
    log_file = open("drift_log.txt", "w")
    print(f"\n[TEST] Starting One-Shot Execution. New arrivals will be FROZEN.")

    crashed_agents = set()
    duration = env_config.get("duration", 10)

    # Map planned actions to steps
    # results[v_id]["actions"] is a list of maneuver strings

    for step in range(duration):
        # A. Apply Planned Actions to Initial Agents
        for v in init_agents:
            if v.id in results:
                actions = results[v.id].get("actions", [])
                if step < len(actions):
                    maneuver = actions[step]
                    # Get low-level control based on CURRENT real state to bridge reality gap
                    # state: [x, y, v, h, lane_idx]
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
                    # Plan finished for this agent, force zero velocity
                    v.speed = 0
                    v.target_speed = 0
                    v.act({"acceleration": 0.0, "steering": 0.0})
            else:
                # No plan for this agent? (shouldn't happen for init_agents)
                v.speed = 0

        # B. Freeze ALL other agents (newly spawned)
        for v in env.unwrapped.road.vehicles:
            if v.id not in init_ids:
                v.speed = 0
                v.target_speed = 0

        # C. Step Simulation
        obs, reward, terminated, truncated, info = env.step(None)

        # D. Reality Check & Telemetry
        for v in init_agents:
            if v.id in results:
                path = results[v.id].get("path", [])
                if (step + 1) < len(path):
                    # --- 1. COLLECT & STORE TELEMETRY (All frames) ---
                    # We log first to ensure the 'crashed' frame is captured
                    planned_state = path[step + 1]
                    real_pos = v.position
                    planned_pos = planned_state[:2]

                    trajectory_real.append(
                        [
                            step,
                            v.id,
                            float(real_pos[0]),
                            float(real_pos[1]),
                            float(v.heading),
                            float(v.speed),
                            bool(v.crashed),
                            float(v.LENGTH),
                            float(v.WIDTH),
                        ]
                    )
                    trajectory_planned.append(
                        [
                            step,
                            v.id,
                            float(planned_pos[0]),
                            float(planned_pos[1]),
                            float(planned_state[3]),
                            float(planned_state[2]),
                            False,
                            float(v.LENGTH),
                            float(v.WIDTH),
                        ]
                    )

                    # --- 2. CRASH AUDIT LOGIC ---
                    if v.crashed:
                        if v.id not in crashed_agents:
                            print(
                                f"\n💥 [GLOBAL CRASH AUDIT] Agent {v.id} at step {step}"
                            )
                            # Forensics...
                            all_road_entities = (
                                env.unwrapped.road.vehicles + env.unwrapped.road.objects
                            )
                            from src.planning.constraints import HighwayConstraints

                            temp_constraints = HighwayConstraints(
                                None, half_l=v.LENGTH / 2, half_w=v.WIDTH / 2
                            )

                            for other in all_road_entities:
                                if other.id == v.id:
                                    continue
                                d = np.linalg.norm(v.position - other.position)
                                if d < 10.0:
                                    overlap_sat = temp_constraints._sat_overlap(
                                        v.position[0],
                                        v.position[1],
                                        v.heading,
                                        other.position[0],
                                        other.position[1],
                                        other.heading,
                                        other.LENGTH / 2,
                                        other.WIDTH / 2,
                                    )
                                    overlap_sim = False
                                    try:
                                        overlap_sim = v._is_colliding(other, dt=0)
                                    except:
                                        pass

                                    if overlap_sat or overlap_sim or d < 2.5:
                                        target_type = (
                                            "VEHICLE"
                                            if other in env.unwrapped.road.vehicles
                                            else "OBJECT"
                                        )
                                        p_rank = priority_rank.get(other.id, "N/A")
                                        print(
                                            f"   >>> COLLISION with {target_type} {other.id} (Rank {p_rank}) | SAT:{overlap_sat} SIM:{overlap_sim} Dist:{d:.3f}m"
                                        )

                            crashed_agents.add(v.id)
                        continue

                    # --- 3. DRIFT LOG (Non-crashed only) ---
                    if v.id not in crashed_agents:
                        drift = np.linalg.norm(real_pos - planned_pos)
                        log_file.write(
                            f"Step {step}, Agent {v.id}, Drift {drift:.4f}, Action {results[v.id]['actions'][step] if step < len(results[v.id]['actions']) else 'NONE'}\n"
                        )

        env.render()

        if debug_mode:
            frame_path = f"debug_plots/steps/step_{step:03d}.png"
            env.unwrapped.save_frame(frame_path)

        time.sleep(0.05)  # Faster playback

        if step % 20 == 0:
            print(f"Step {step}/{duration} | Active Agents: {len(init_agents)}")

        if terminated or truncated:
            break

    if debug_mode:
        # Save all trajectory data to CSV
        import csv

        with open("debug_plots/trajectories_real.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "agent_id",
                    "x",
                    "y",
                    "heading",
                    "speed",
                    "crashed",
                    "length",
                    "width",
                ]
            )
            writer.writerows(trajectory_real)

        with open("debug_plots/trajectories_planned.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "agent_id",
                    "x",
                    "y",
                    "heading",
                    "speed",
                    "crashed",
                    "length",
                    "width",
                ]
            )
            writer.writerows(trajectory_planned)

        print(f"\n✅ [DEBUG] Trajectories saved to CSV (real/planned)")

    # 6. Drift Log Summary
    with open("drift_log.txt", "r") as f:
        logs = f.readlines()
        if logs:
            print("\n📊 [DRIFT SUMMARY] Last 5 entries:")
            for line in logs[-5:]:
                print(f"   {line.strip()}")

    print("\n[TEST] Execution Completed.")
    env.close()


if __name__ == "__main__":
    test_primitives_one_shot()
