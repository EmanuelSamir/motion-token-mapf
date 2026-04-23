import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import torch
import os
import pandas as pd
import imageio
from typing import List, Dict, Any
from collections import deque

from src.env.merge_interaction_env import MergeInteractionEnv, CAVVehicle
from src.planning.motion_primitives import MotionPrimitives
from src.planning.pbs_planner import PBSPlanner
from src.planning.bk_planner import BKPlanner


class BenchmarkSuite:
    """
    Unified benchmarking for multi-agent coordination methods.
    Evaluates: BK-PBS, Baseline-PBS (Static/Const), and Default IDM.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.env_config = OmegaConf.to_container(cfg.env, resolve=True)
        self.env_config["cav_ratio"] = cfg.get("test_cav_ratio", 0.5)
        self.env_config["target_vehicles_count"] = 20  # Sync with stress tests

        # Unified Render Mode: 'none', 'human', 'record'
        self.render_mode = cfg.get("test_render_mode", "none")
        if self.render_mode == "record":
            self.env_config["render_mode"] = "rgb_array"
        elif self.render_mode == "human":
            self.env_config["render_mode"] = "human"
        else:
            self.env_config["render_mode"] = None

        self.num_rollouts = cfg.get("num_rollouts", 10)
        self.duration = cfg.get("test_duration", 300)  # steps

        # Calculate goal_x dynamically from environment geometry
        total_len = (
            self.env_config.get("ramp_distance", 150)
            + self.env_config.get("merging_length", 80)
            + self.env_config.get("post_merge_length", 200)
        )
        self.goal_x = total_len - 30.0  # Goal is 30m before the absolute end

        self.output_dir = "outputs/benchmarks"
        self.video_dir = os.path.join(self.output_dir, "videos")
        os.makedirs(self.output_dir, exist_ok=True)
        if self.render_mode == "record":
            os.makedirs(self.video_dir, exist_ok=True)

        # Load weights for BK-PBS
        self.ckpt_path = cfg.get("ckpt_path", "checkpoints/latest.ckpt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detailed_results = []

    def run_full_benchmark(self):
        method_variants = [
            ("IDM-Only", "none"),
            ("Baseline-PBS", "static"),
            ("Baseline-PBS", "constant_vel"),
            ("BK-PBS", "reactive"),
        ]

        # Filter methods based on config if provided
        selected_methods = self.cfg.get("methods", ["all"])
        if "all" not in selected_methods:
            method_variants = [
                m
                for m in method_variants
                if m[0].lower().replace("-", "_") in selected_methods
                or m[1] in selected_methods
            ]

        all_results = []
        for name, variant in method_variants:
            print(f"\n🚀 [BENCHMARK] Evaluating Method: {name} ({variant})")
            method_metrics = self._run_method_rollouts(name, variant)
            all_results.append(method_metrics)

        # Summary Report
        df_summary = pd.DataFrame(all_results)

        # Consistent mapping for renaming
        column_mapping = {
            "success_rate": "Success Rate (Merge)",
            "collision_rate": "Collision Rate",
            "avg_time_to_goal": "Time to Finish Task",
            "avg_latency": "Avg. Cycle Latency",
        }
        df_summary_disp = df_summary.rename(columns=column_mapping)

        df_summary.to_csv(f"{self.output_dir}/summary.csv", index=False)
        df_detailed = pd.DataFrame(self.detailed_results)
        df_detailed.to_csv(f"{self.output_dir}/detailed_results.csv", index=False)

        print("\n✅ [BENCHMARK COMPLETE]")
        print(f"   Summary saved to: {self.output_dir}/summary.csv")
        print("\nSummary Table:")
        display_cols = [
            "method",
            "variant",
            "Success Rate (Merge)",
            "Collision Rate",
            "Time to Finish Task",
            "Avg. Cycle Latency",
        ]

        # Ensure columns exist for display
        for col in display_cols:
            if col not in df_summary_disp.columns:
                df_summary_disp[col] = 0.0

        print(df_summary_disp[display_cols])

    def _run_method_rollouts(self, name: str, variant: str) -> Dict:
        rollout_metrics = []
        for i in range(self.num_rollouts):
            seed = 42 + i
            metrics = self._single_rollout(name, variant, seed)

            detailed_entry = {
                "method": name,
                "variant": variant,
                "seed": seed,
                **metrics,
            }
            self.detailed_results.append(detailed_entry)
            rollout_metrics.append(metrics)

            print(
                f"   Rollout {i + 1}/{self.num_rollouts} | Collision Rate: {metrics['collision_rate']:.1%} | Success: {metrics['success_rate']:.1%}"
            )

        def safe_mean(lst):
            return np.mean(lst) if len(lst) > 0 else 0.0

        return {
            "method": name,
            "variant": variant,
            "success_rate": safe_mean([m["success_rate"] for m in rollout_metrics]),
            "collision_rate": safe_mean([m["collision_rate"] for m in rollout_metrics]),
            "avg_time_to_goal": safe_mean(
                [
                    m["avg_time_to_goal"]
                    for m in rollout_metrics
                    if m["avg_time_to_goal"] > 0
                ]
            ),
            "avg_latency": safe_mean([m["avg_latency"] for m in rollout_metrics]),
            "cav_hdv_collisions": np.sum(
                [m["cav_hdv_collisions"] for m in rollout_metrics]
            ),
        }

    def _update_rollout_metrics(
        self,
        step,
        env,
        cav_start_times,
        cav_finish_times,
        last_known_positions,
        crashed_pairs,
        agent_types,
    ):
        # 1. Goal Check (Active agents)
        for v in env.unwrapped.road.vehicles:
            last_known_positions[v.id] = v.position.copy()
            if getattr(v, "is_cav", False) and v.id not in cav_finish_times:
                if v.position[0] >= self.goal_x:
                    cav_finish_times[v.id] = step

        # 2. Removal Events Check (Graduations vs Crashes)
        removal_events = getattr(env.unwrapped, "latest_removal_events", [])
        for event in removal_events:
            v_id = event["id"]
            if event["is_cav"]:
                if event["reason"] == "graduated" and v_id not in cav_finish_times:
                    cav_finish_times[v_id] = step

            if event["reason"] == "crashed" and v_id not in [
                p[0] for p in crashed_pairs
            ] + [p[1] for p in crashed_pairs]:
                # Record crash pairs for statistics
                for other in (
                    env.unwrapped.road.vehicles
                ):  # This might miss the 'other' if both were removed
                    if np.linalg.norm(np.array([event["x"], 0]) - other.position) < 5.0:
                        crashed_pairs.add(tuple(sorted((v_id, other.id))))
                if not any(v_id in p for p in crashed_pairs):
                    crashed_pairs.add(
                        (v_id, -1)
                    )  # Single vehicle crash or unknown other

    def _single_rollout(self, name: str, variant: str, seed: int) -> Dict:
        env_config = self.env_config.copy()
        env_config["collision_terminal"] = False
        env_config.update({"offroad_terminal": False})

        env = MergeInteractionEnv(env_config)
        env.reset(seed=seed)
        mp = MotionPrimitives(road=env.unwrapped.road)

        planner = None
        if name == "BK-PBS":
            planner = BKPlanner(
                mp,
                self.ckpt_path,
                device=self.device,
                horizon_agg=150.0,
                horizon_safe=80.0,
            )
        elif name == "Baseline-PBS":
            planner = PBSPlanner(mp, horizon_agg=150.0, horizon_safe=80.0)
            planner.set_baseline_mode(variant)

        # Tracking states
        crashed_pairs, cav_start_times, cav_finish_times = set(), {}, {}
        agent_types, agents_history, last_known_positions = {}, {}, {}
        full_trajectories = {}  # v_id -> list of (step, x)
        current_plans, latency_records = {}, []
        prev_num_vehicles, planning_freq = 0, 10
        frames = []

        for step in range(self.duration):
            # A. Maintenance & Registration
            for v in env.unwrapped.road.vehicles:
                if v.id not in agent_types:
                    agent_types[v.id] = "CAV" if getattr(v, "is_cav", False) else "HDV"
                    if agent_types[v.id] == "CAV":
                        cav_start_times[v.id] = step

                if v.id not in agents_history:
                    agents_history[v.id] = deque(maxlen=10)
                vx, vy = v.speed * np.cos(v.heading), v.speed * np.sin(v.heading)
                agents_history[v.id].append(
                    [v.position[0], v.position[1], vx, vy, v.heading]
                )

                # Store full history for diagnostics
                if v.id not in full_trajectories:
                    full_trajectories[v.id] = []
                full_trajectories[v.id].append((step, v.position[0]))

            # B. Periodic Planning
            if name != "IDM-Only":
                num_v = len(env.unwrapped.road.vehicles)
                plan_exhausted = any(
                    agent_types.get(vid) == "CAV"
                    and (step - getattr(self, f"last_plan_step_{vid}", 0))
                    >= len(p.get("actions", [])) - 2
                    for vid, p in current_plans.items()
                )

                if planner and (
                    (step % planning_freq == 0)
                    or (num_v != prev_num_vehicles)
                    or plan_exhausted
                ):
                    prev_num_vehicles = num_v
                    planner.set_agent_types(agent_types)
                    all_active = [
                        v
                        for v in env.unwrapped.road.vehicles
                        if v.position[0] < self.goal_x + 20 and not v.crashed
                    ]
                    if name == "BK-PBS" and len(all_active) > 10:
                        all_active = sorted(
                            all_active, key=lambda v: v.position[0], reverse=True
                        )[:10]

                    active_states, v_info, active_hist = {}, [], {}
                    for v in all_active:
                        active_states[v.id] = np.array(
                            [v.position[0], v.position[1], v.speed, v.heading, 0]
                        )
                        v_info.append(
                            {"id": v.id, "length": v.LENGTH, "width": v.WIDTH}
                        )
                        active_hist[v.id] = list(agents_history[v.id])

                    t0 = time.time()
                    current_plans = planner.plan_all(
                        active_states, v_info=v_info, agents_history=active_hist
                    )
                    latency_records.append(time.time() - t0)
                    for vid in current_plans:
                        setattr(self, f"last_plan_step_{vid}", step)

                # Action Application
                for vid, res in current_plans.items():
                    if agent_types.get(vid) == "CAV" and "actions" in res:
                        p_step = step - getattr(self, f"last_plan_step_{vid}", 0)
                        if p_step < len(res["actions"]):
                            v = next(
                                (
                                    veh
                                    for veh in env.unwrapped.road.vehicles
                                    if veh.id == vid
                                ),
                                None,
                            )
                            if v:
                                state = np.array(
                                    [
                                        v.position[0],
                                        v.position[1],
                                        v.speed,
                                        v.heading,
                                        0,
                                    ]
                                )
                                v.act(
                                    mp.get_low_level_action(
                                        state, res["actions"][p_step]
                                    )
                                )

            # C. Step & Render
            env.step(None)
            if self.render_mode == "human":
                env.render()
            elif self.render_mode == "record":
                f = env.render()
                if f is not None:
                    frames.append(f)

            # D. Metrics
            self._update_rollout_metrics(
                step,
                env,
                cav_start_times,
                cav_finish_times,
                last_known_positions,
                crashed_pairs,
                agent_types,
            )

            # Clean periodic print
            if step % 100 == 0:
                lead_x = max([v.position[0] for v in env.unwrapped.road.vehicles] + [0])
                print(
                    f"      [Step {step:3}] Lead Agent at {lead_x:5.1f}m / {self.goal_x:5.1f}m"
                )

        # Summarize rollout
        cav_hdv, hdv_hdv, cav_cav = 0, 0, 0
        for ida, idb in crashed_pairs:
            ta, tb = agent_types.get(ida), agent_types.get(idb)
            if ta == "CAV" and tb == "CAV":
                cav_cav += 1
            elif ta == "HDV" and tb == "HDV":
                hdv_hdv += 1
            else:
                cav_hdv += 1

        goal_times = [
            cav_finish_times[vid] - cav_start_times[vid] for vid in cav_finish_times
        ]
        res = {
            "collision_rate": len(crashed_pairs) / max(1, len(agent_types)),
            "cav_hdv_collisions": cav_hdv,
            "hdv_hdv_collisions": hdv_hdv,
            "cav_cav_collisions": cav_cav,
            "avg_time_to_goal": np.mean(goal_times) if goal_times else 0.0,
            "avg_latency": np.mean(latency_records) if latency_records else 0.0,
            "success_rate": len(cav_finish_times) / max(1, len(cav_start_times)),
        }
        env.close()

        if self.render_mode == "record" and frames:
            path = os.path.join(
                self.video_dir, f"{name.lower()}_{variant}", f"seed_{seed}.mp4"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            imageio.mimsave(path, frames, fps=15)

        return res


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    suite = BenchmarkSuite(cfg)
    suite.run_full_benchmark()


if __name__ == "__main__":
    main()
