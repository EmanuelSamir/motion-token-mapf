import numpy as np
import torch

class MotionTokenizer:
    """
    Implements the discrete motion tokenization scheme from MotionLM.
    Uses uniform quantization (13 bins as per paper), Verlet-wrapping, and Greedy Search.
    
    Optimized for physical traffic: 1.0 peak delta-delta.
    """
class MotionTokenizer:
    """
    Implements the full MotionLM discretization:
    1. Velocity Grid (128 bins, -8m to 15m)
    2. Delta-Delta tokens (13 bins, index offsets -6 to +6)
    
    This ensures Token 84 (offset 0,0) has ZERO drift for constant velocity.
    """
    def __init__(self, num_bins=13, velocity_bins=128, max_velocity_x=18.0, max_velocity_y=8.0):
        self.num_bins = num_bins # 13
        self.v_bins_count = velocity_bins # 128
        self.max_v_x = max_velocity_x
        self.max_v_y = max_velocity_y
        
        # Velocity Grids (Asymmetric)
        self.v_grid_x = np.linspace(-max_velocity_x, max_velocity_x, velocity_bins)
        self.v_grid_y = np.linspace(-max_velocity_y, max_velocity_y, velocity_bins)
        
        self.v_step_x = self.v_grid_x[1] - self.v_grid_x[0]
        self.v_step_y = self.v_grid_y[1] - self.v_grid_y[0]
        
        # Acceleration Offsets (Index shifts: -6 to +6)
        self.offset_range = num_bins // 2
        self.offsets = np.arange(-self.offset_range, self.offset_range + 1)
        
        self.vocab_size = num_bins * num_bins
        self.INVALID_TOKEN = 169

    def _get_velocity_bin(self, v):
        """ Maps continuous velocity to nearest grid indices [idx_x, idx_y] """
        idx_x = np.round((v[0] + self.max_v_x) / self.v_step_x).astype(int)
        idx_y = np.round((v[1] + self.max_v_y) / self.v_step_y).astype(int)
        return np.array([
            np.clip(idx_x, 0, self.v_bins_count - 1),
            np.clip(idx_y, 0, self.v_bins_count - 1)
        ])

    def decode_token(self, token, prev_delta_bin_idx):
        """
        In Paper: New_Vel_Bin = Prev_Vel_Bin + Token_Offset
        """
        if token >= self.vocab_size:
            return 0.0, 0.0, prev_delta_bin_idx
            
        off_idx_x = token // self.num_bins
        off_idx_y = token % self.num_bins
        
        offset_x = self.offsets[off_idx_x]
        offset_y = self.offsets[off_idx_y]
        
        new_bin_x = np.clip(prev_delta_bin_idx[0] + offset_x, 0, self.v_bins_count - 1)
        new_bin_y = np.clip(prev_delta_bin_idx[1] + offset_y, 0, self.v_bins_count - 1)
        
        new_v_x = self.v_grid_x[new_bin_x]
        new_v_y = self.v_grid_y[new_bin_y]
        
        return new_v_x, new_v_y, np.array([new_bin_x, new_bin_y])

    def tokenize_trajectory(self, trajectory, initial_delta):
        T = len(trajectory)
        if T < 2: return []
        
        traj_xy = np.array(trajectory)[:, :2]
        curr_recon_pos = traj_xy[0].copy()
        
        # Initial velocity bin
        prev_v_bin = self._get_velocity_bin(initial_delta)
        
        # Precompute all possible candidate velocity bins [169, 2]
        # These are constant for every step: prev_v_bin + offsets
        ox, oy = np.meshgrid(self.offsets, self.offsets, indexing='ij')
        token_offsets = np.stack([ox.flatten(), oy.flatten()], axis=1) # [169, 2]
        
        tokens = []
        for t in range(1, T):
            target_pos = traj_xy[t]
            
            # Vectorized candidate bins
            candidate_bins = np.clip(prev_v_bin + token_offsets, 0, self.v_bins_count - 1)
            
            # Independent grids for X and Y
            candidate_v_x = self.v_grid_x[candidate_bins[:, 0]]
            candidate_v_y = self.v_grid_y[candidate_bins[:, 1]]
            candidate_v = np.stack([candidate_v_x, candidate_v_y], axis=1) # [169, 2]
            
            candidate_pos = curr_recon_pos + candidate_v # [169, 2]
            
            # Squared error [169]
            errors = np.sum((candidate_pos - target_pos)**2, axis=1)
            best_token = np.argmin(errors)
            
            tokens.append(int(best_token))
            prev_v_bin = candidate_bins[best_token]
            curr_recon_pos = curr_recon_pos + np.array([self.v_grid_x[prev_v_bin[0]], self.v_grid_y[prev_v_bin[1]]])
            
        return tokens

    def reconstruct_trajectory(self, start_pos, tokens, initial_delta):
        curr_pos = np.array(start_pos, dtype=np.float32)[:2]
        trajectory = [curr_pos.copy()]
        
        prev_v_bin = self._get_velocity_bin(initial_delta)
        
        for token in tokens:
            if token >= self.vocab_size:
                # If invalid, treat as 'Idle' (maintain velocity) but don't stop
                v_x, v_y, prev_v_bin = self.v_grid_x[prev_v_bin[0]], self.v_grid_y[prev_v_bin[1]], prev_v_bin
            else:
                v_x, v_y, prev_v_bin = self.decode_token(token, prev_v_bin)
                
            curr_pos += np.array([v_x, v_y])
            trajectory.append(curr_pos.copy())
            
        return np.array(trajectory)
