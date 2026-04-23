import numpy as np
import torch
from src.planning.pbs_planner import PBSPlanner, HighLevelNode
from src.models.inference import MotionLMInferenceEngine
from src.utils.tokenizer import MotionTokenizer


class BKPlanner(PBSPlanner):
    """
    Behavior Prediction Priority-Based Search (BK-PBS).
    Inherits from PBSPlanner and uses MotionLM for conditional HDV forecasting.
    """

    def __init__(self, mp, model_path, device="cpu", **kwargs):
        super().__init__(mp, **kwargs)
        self.engine = MotionLMInferenceEngine(model_path, device=device)
        self.tokenizer = self.engine.tokenizer
        self.agent_types = {}  # v_id -> "CAV" or "HDV"

    def set_agent_types(self, agent_types):
        self.agent_types = agent_types

    def _plan_with_constraints(
        self,
        agents_states,
        constraints,
        prev_results=None,
        affected_agent=-1,
        base_order=None,
        agents_history=None,
    ):
        """
        BK-PBS version: Plans CAVs with A* and HDVs with MotionLM conditioning.
        """
        v_ids = list(agents_states.keys())
        if base_order is None:
            base_order = self.get_priority_order(agents_states)

        order = self._get_topological_order(v_ids, constraints, base_order)

        results = {}
        planned_paths = {}  # agent_id -> list of np.ndarray

        # Prepare history context for MotionLM (all agents)
        # Build history tensor [N, 10, 5] -> Expected format: [x, y, vx, vy, yaw]
        history_list = []
        id_to_idx = {}
        for idx, vid in enumerate(v_ids):
            state = agents_states[vid]
            # test script state is [x, y, speed, h, lane]
            x, y, speed, h, lane = state
            vx = speed * np.cos(h)
            vy = speed * np.sin(h)

            if (
                agents_history is not None
                and vid in agents_history
                and len(agents_history[vid]) > 0
            ):
                # Use actual provided history from simulator
                hist_agent = list(agents_history[vid])
            else:
                # Start with at least the current frame if no history available
                hist_agent = [[x, y, vx, vy, h]]

            # Ensure sequence is exactly 10 states long
            # (Truncate older states or pad by extrapolating backwards)
            if len(hist_agent) > 10:
                hist_agent = hist_agent[-10:]
            elif len(hist_agent) < 10:
                missing = 10 - len(hist_agent)
                oldest_state = hist_agent[0]
                old_x, old_y, old_vx, old_vy, old_h = oldest_state

                padding = []
                for t_step in range(-missing, 0):
                    past_x = old_x + old_vx * t_step * self.dt
                    past_y = old_y + old_vy * t_step * self.dt
                    padding.append([past_x, past_y, old_vx, old_vy, old_h])

                hist_agent = padding + hist_agent

            history_list.append(hist_agent)
            id_to_idx[vid] = idx

        history_tensor = torch.tensor(
            np.array(history_list), dtype=torch.float32
        )  # [N, 10, 5]

        for v_id in order:
            v_type = self.agent_types.get(v_id, "HDV")
            state = agents_states[v_id]

            if v_type == "CAV":
                # Standard A* planning for CAVs
                res = self.astar.plan(
                    state,
                    self.constraints_evaluator,
                    obstacles=planned_paths,
                    v_id=v_id,
                )
                if not res.get("path"):
                    return None
                results[v_id] = res
                planned_paths[v_id] = res["path"]
            else:
                # CONDITIONAL PREDICTION for HDVs
                # 1. Identify higher-priority CAV plans for clamping
                clamped_tokens = {}
                for other_id, other_path in planned_paths.items():
                    if self.agent_types.get(other_id) == "CAV":
                        # Convert A* path to tokens
                        # We need the initial_delta to match the tokenizer's Verlet logic
                        # Simplified: initial_delta is the velocity vector at start
                        v0 = agents_states[other_id][2]
                        h0 = agents_states[other_id][3]
                        init_delta = (
                            np.array([v0 * np.cos(h0), v0 * np.sin(h0)]) * self.dt
                        )

                        tokens = self.tokenizer.tokenize_trajectory(
                            other_path, initial_delta=init_delta
                        )
                        clamped_tokens[id_to_idx[other_id]] = tokens

                # 2. Run Inference Engine
                # This returns tokens for ALL agents, but we only care about v_id
                pred_dict = self.engine.predict_joint_v2(
                    history_tensor,
                    clamped_tokens=clamped_tokens,
                    max_timesteps=20,  # Matches trajectory horizon
                )

                agent_tokens = pred_dict[id_to_idx[v_id]]

                # 3. Reconstruct physical trajectory for collision checking
                v0_self = agents_states[v_id][2]
                h0_self = agents_states[v_id][3]
                init_delta_self = (
                    np.array([v0_self * np.cos(h0_self), v0_self * np.sin(h0_self)])
                    * self.dt
                )

                path_recon = self.tokenizer.reconstruct_trajectory(
                    agents_states[v_id][:2], agent_tokens, initial_delta=init_delta_self
                )

                # Convert back to full state format (x, y, v, h, lane)
                # (Simplification: v and h calculated from diffs)
                full_path = []
                for k in range(len(path_recon)):
                    # Filling in dummy v, h, lane for collision evaluator compatibility
                    # Ideal: calculate real v, h
                    full_path.append(
                        np.array(
                            [path_recon[k][0], path_recon[k][1], v0_self, h0_self, 0]
                        )
                    )

                results[v_id] = {"path": full_path, "type": "predicted"}
                planned_paths[v_id] = full_path

        return results
