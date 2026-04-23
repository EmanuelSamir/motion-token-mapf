import numpy as np
import torch

__all__ = ["MotionLMVocabulary"]

class MotionLMVocabulary:
    """
    Handles discretization and reconstruction of motion tokens for MotionLM.
    
    Research Alignment (Section 3.2.2 & Appendix A):
    - Base Delta Action Space: Uniformly quantized [-18.0m, 18.0m] with 128 bins.
    - Verlet-wrapped Action Space: 13 tokens per coordinate (169 total).
    - Token 0 (Zero Action) indicates constant velocity (repeating the previous delta index).
    - Greedy Search: Sequentially selects actions to minimize reconstruction error.
    """

    def __init__(self, max_delta=18.0, num_delta_bins=128, bins_per_axis=13):
        self.max_delta = max_delta
        self.num_delta_bins = num_delta_bins
        self.bins_per_axis = bins_per_axis
        self.vocab_size = bins_per_axis**2
        
        # Delta space parameters
        self.delta_min = -max_delta
        self.delta_max = max_delta
        # For 128 bins, we use index 64 as exactly 0.0
        self.delta_step = max_delta / 64 # 0.28125 for 18m
        
        # Verlet space parameters (13 tokens -> offsets -6 to +6)
        self.verlet_center = bins_per_axis // 2  # 6
        self.joint_center_token = self.verlet_center * bins_per_axis + self.verlet_center # 84

    def _map_axis_to_vocab(self, bin_idx):
        """Maps a single-axis bin (0-12) to a token (0-12).
        Ensures bin 6 (zero offset) maps to Token 0.
        """
        if bin_idx == self.verlet_center: # 6
            return 0
        if bin_idx == 0:
            return self.verlet_center
        return bin_idx

    def _map_vocab_to_axis(self, token):
        """Inverts the axis mapping."""
        if token == 0:
            return self.verlet_center
        if token == self.verlet_center:
            return 0
        return token

    def _val_to_delta_idx(self, val):
        """Discretizes a continuous delta value to the closest bin index [0, 127]."""
        idx = np.round((val - self.delta_min) / self.delta_step).astype(np.int32)
        return np.clip(idx, 0, self.num_delta_bins - 1)

    def _delta_idx_to_val(self, idx):
        """Converts a bin index back to the continuous value (exact representation if possible)."""
        return self.delta_min + idx * self.delta_step

    def quantize_deltas_verlet(self, future_waypoints, pos_t0, pos_t_minus_1, **kwargs):
        """
        Iterative 'Greedy Search' for quantization (Section 3.2.2).
        Returns separate tokens for X and Y for each timestep.
        
        Returns:
            tokens: [..., T, 2] indices in [0, 12].
        """
        if torch.is_tensor(future_waypoints):
            future_waypoints = future_waypoints.detach().cpu().numpy()
        if torch.is_tensor(pos_t0):
            pos_t0 = pos_t0.detach().cpu().numpy()
        if torch.is_tensor(pos_t_minus_1):
            pos_t_minus_1 = pos_t_minus_1.detach().cpu().numpy()

        orig_shape = future_waypoints.shape[:-2]
        num_steps = future_waypoints.shape[-2]
        
        # Flatten for processing
        future_waypoints = future_waypoints.reshape(-1, num_steps, 2)
        pos_t0 = pos_t0.reshape(-1, 2)
        pos_t_minus_1 = pos_t_minus_1.reshape(-1, 2)
        
        num_envs = future_waypoints.shape[0]
        tokens = np.zeros((num_envs, num_steps, 2), dtype=np.int32)
        
        # Initial state
        curr_pos = pos_t0.copy()
        # Compute initial delta index (v_0 = pos_0 - pos_{-1})
        init_delta = pos_t0 - pos_t_minus_1
        curr_idx_x = self._val_to_delta_idx(init_delta[..., 0])
        curr_idx_y = self._val_to_delta_idx(init_delta[..., 1])

        for t in range(num_steps):
            target_pos = future_waypoints[:, t, :]
            
            # Find best offset for X
            target_delta_x = target_pos[:, 0] - curr_pos[:, 0]
            target_idx_x = self._val_to_delta_idx(target_delta_x)
            offset_x = target_idx_x - curr_idx_x
            ax = np.clip(offset_x + self.verlet_center, 0, self.bins_per_axis - 1).astype(np.int32)
            
            # Find best offset for Y
            target_delta_y = target_pos[:, 1] - curr_pos[:, 1]
            target_idx_y = self._val_to_delta_idx(target_delta_y)
            offset_y = target_idx_y - curr_idx_y
            ay = np.clip(offset_y + self.verlet_center, 0, self.bins_per_axis - 1).astype(np.int32)
            
            # Record separate tokens
            for i in range(num_envs):
                tokens[i, t, 0] = self._map_axis_to_vocab(ax[i])
                tokens[i, t, 1] = self._map_axis_to_vocab(ay[i])
            
            # Update reconstructed state
            best_offset_x = ax - self.verlet_center
            best_offset_y = ay - self.verlet_center
            
            curr_idx_x = np.clip(curr_idx_x + best_offset_x, 0, self.num_delta_bins - 1)
            curr_idx_y = np.clip(curr_idx_y + best_offset_y, 0, self.num_delta_bins - 1)
            
            recon_delta_x = self._delta_idx_to_val(curr_idx_x)
            recon_delta_y = self._delta_idx_to_val(curr_idx_y)
            
            curr_pos[:, 0] += recon_delta_x
            curr_pos[:, 1] += recon_delta_y
            
        return tokens.reshape(*orig_shape, num_steps, 2)

    def reconstruct(self, tokens, pos_t0, pos_t_minus_1, **kwargs):
        """
        Inverts the discretization to reconstruct coordinates.
        Expects tokens of shape [..., T, 2].
        """
        if torch.is_tensor(tokens):
            tokens = tokens.detach().cpu().numpy()
        if torch.is_tensor(pos_t0):
            pos_t0 = pos_t0.detach().cpu().numpy()
        if torch.is_tensor(pos_t_minus_1):
            pos_t_minus_1 = pos_t_minus_1.detach().cpu().numpy()

        orig_shape = tokens.shape[:-2]
        num_steps = tokens.shape[-2]
        
        # Flatten
        tokens = tokens.reshape(-1, num_steps, 2)
        pos_t0 = pos_t0.reshape(-1, 2)
        pos_t_minus_1 = pos_t_minus_1.reshape(-1, 2)
        
        num_envs = tokens.shape[0]
        reconstructed = np.zeros((num_envs, num_steps, 2), dtype=np.float32)
        
        curr_pos = pos_t0.copy()
        init_delta = pos_t0 - pos_t_minus_1
        curr_idx_x = self._val_to_delta_idx(init_delta[..., 0])
        curr_idx_y = self._val_to_delta_idx(init_delta[..., 1])

        for t in range(num_steps):
            step_tokens = tokens[:, t, :] # [num_envs, 2]
            
            for i in range(num_envs):
                ax = self._map_vocab_to_axis(step_tokens[i, 0])
                ay = self._map_vocab_to_axis(step_tokens[i, 1])
                
                offset_x = ax - self.verlet_center
                offset_y = ay - self.verlet_center
                
                curr_idx_x[i] = np.clip(curr_idx_x[i] + offset_x, 0, self.num_delta_bins - 1)
                curr_idx_y[i] = np.clip(curr_idx_y[i] + offset_y, 0, self.num_delta_bins - 1)
                
                dx = self._delta_idx_to_val(curr_idx_x[i])
                dy = self._delta_idx_to_val(curr_idx_y[i])
                
                curr_pos[i, 0] += dx
                curr_pos[i, 1] += dy
                
                reconstructed[i, t, 0] = curr_pos[i, 0]
                reconstructed[i, t, 1] = curr_pos[i, 1]

        return torch.from_numpy(reconstructed.reshape(*orig_shape, num_steps, 2)).float()
