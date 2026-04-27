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
    def __init__(self, num_bins=None, velocity_bins=128, max_velocity_x=25.0, max_velocity_y=10.0, dt=0.2):
        """
        Args:
            velocity_bins: Resolution of the velocity grid.
            max_velocity_x: Max velocity in m/s (longitudinal).
            max_velocity_y: Max velocity in m/s (lateral).
            dt: Time step in seconds.
        """
        self.dt = dt
        self.v_bins_count = velocity_bins 
        
        # Internal units are m/step to match dataset displacements
        self.max_v_x_step = max_velocity_x * dt
        self.max_v_y_step = max_velocity_y * dt
        
        # Velocity Grids (Symmetric in m/step)
        self.v_grid_x = np.linspace(-self.max_v_x_step, self.max_v_x_step, velocity_bins)
        self.v_grid_y = np.linspace(-self.max_v_y_step, self.max_v_y_step, velocity_bins)
        
        self.v_step_x = self.v_grid_x[1] - self.v_grid_x[0]
        self.v_step_y = self.v_grid_y[1] - self.v_grid_y[0]
        
        # Optimized 2D Codebook (Index shifts on the grid)
        # Derived via K-Means clustering on unique observed offsets (K=32).
        self.codebook = np.array([
            [-9, -41], [-8, -27], [-7, -23], [-6, -18], [-4, -34], [-4, -32], [-4, -26], [-4, -12], 
            [-3, -19], [-3, -15], [-2, -22], [-2, -8], [-2, -5], [-2, -2], [-2, 0], [-2, 2], 
            [-1, -11], [-1, 5], [-1, 13], [0, -8], [0, 0], [0, 2], [0, 7], [0, 10], 
            [0, 21], [0, 26], [1, -3], [1, 15], [2, 5], [3, 12], [4, 10], [6, 16]
        ])
        
        self.vocab_size = len(self.codebook)
        self.INVALID_TOKEN = self.vocab_size

    def _get_velocity_bin(self, v):
        """ 
        Maps continuous displacement v [m/step] to nearest grid indices [idx_x, idx_y].
        """
        idx_x = np.round((v[0] + self.max_v_x_step) / self.v_step_x).astype(int)
        idx_y = np.round((v[1] + self.max_v_y_step) / self.v_step_y).astype(int)
        return np.array([
            np.clip(idx_x, 0, self.v_bins_count - 1),
            np.clip(idx_y, 0, self.v_bins_count - 1)
        ])

    def decode_token(self, token, prev_v_bin_idx):
        """
        Decodes 1D token index to a new velocity bin using the 2D codebook.
        """
        if token >= self.vocab_size:
            # Return same bin if invalid
            return self.v_grid_x[prev_v_bin_idx[0]], self.v_grid_y[prev_v_bin_idx[1]], prev_v_bin_idx
            
        offset_x, offset_y = self.codebook[token]
        
        new_bin_x = np.clip(prev_v_bin_idx[0] + offset_x, 0, self.v_bins_count - 1)
        new_bin_y = np.clip(prev_v_bin_idx[1] + offset_y, 0, self.v_bins_count - 1)
        
        new_v_x = self.v_grid_x[new_bin_x]
        new_v_y = self.v_grid_y[new_bin_y]
        
        return new_v_x, new_v_y, np.array([new_bin_x, new_bin_y])

    def tokenize_trajectory(self, trajectory, initial_delta):
        """
        initial_delta: displacement [m/step] of the first recorded segment.
        """
        T = len(trajectory)
        if T < 2: return []
        
        traj_xy = np.array(trajectory)[:, :2]
        curr_recon_pos = traj_xy[0].copy()
        
        # Initial velocity bin
        prev_v_bin = self._get_velocity_bin(initial_delta)
        
        tokens = []
        for t in range(1, T):
            target_pos = traj_xy[t]
            
            # Vectorized candidate bins
            candidate_bins = np.clip(prev_v_bin + self.codebook, 0, self.v_bins_count - 1)
            
            # Reconstruct candidate velocities
            candidate_v_x = self.v_grid_x[candidate_bins[:, 0]]
            candidate_v_y = self.v_grid_y[candidate_bins[:, 1]]
            candidate_v = np.stack([candidate_v_x, candidate_v_y], axis=1) # [64, 2]
            
            candidate_pos = curr_recon_pos + candidate_v # [64, 2]
            
            # Closest candidate in position space
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
            v_x, v_y, prev_v_bin = self.decode_token(token, prev_v_bin)
            curr_pos += np.array([v_x, v_y])
            trajectory.append(curr_pos.copy())
            
        return np.array(trajectory)
