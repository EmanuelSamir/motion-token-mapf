import numpy as np
import torch

class MotionTokenizer:
    """
    Implements the discrete motion tokenization scheme from MotionLM.
    Uses uniform quantization (13 bins as per paper), Verlet-wrapping, and Greedy Search.
    
    Optimized for physical traffic: 1.0 peak delta-delta.
    """
    def __init__(self, num_bins=13, max_delta_delta=1.0):
        """
        Args:
            num_bins: Number of bins per coordinate (default 13 -> 169 total).
            max_delta_delta: Maximum change in delta per step.
                             1.0 covers 5m/s^2 acceleration at 5Hz.
        """
        self.num_bins = num_bins
        self.max_dd = max_delta_delta
        self.bins = np.linspace(-max_delta_delta, max_delta_delta, num_bins)
        self.vocab_size = num_bins * num_bins
        self.INVALID_TOKEN = 170
        
        # Precompute all possible delta-deltas for greedy search
        self.all_dd = []
        for i in range(num_bins):
            for j in range(num_bins):
                self.all_dd.append([self.bins[i], self.bins[j]])
        self.all_dd = np.array(self.all_dd) # [169, 2]

    def decode_token(self, token):
        idx_x = token // self.num_bins
        idx_y = token % self.num_bins
        return self.bins[idx_x], self.bins[idx_y]

    def tokenize_trajectory(self, trajectory, initial_delta=None):
        T = len(trajectory)
        if T < 2: return []
        
        # Ensure we only use x, y for tokenization math
        traj_xy = np.array(trajectory)[:, :2]

        curr_recon_pos = traj_xy[0].copy()
        prev_recon_delta = np.array(initial_delta if initial_delta is not None else [0.0, 0.0], dtype=np.float32)
        
        tokens = []
        for t in range(1, T):
            target_pos = traj_xy[t]
            
            # Vectorized search
            possible_next_deltas = prev_recon_delta + self.all_dd # [169, 2]
            possible_next_pos = curr_recon_pos + possible_next_deltas # [169, 2]
            
            errors = np.sum((possible_next_pos - target_pos)**2, axis=1)
            best_token = np.argmin(errors)
            
            tokens.append(int(best_token))
            
            best_dd = self.all_dd[best_token]
            prev_recon_delta = prev_recon_delta + best_dd
            curr_recon_pos = curr_recon_pos + prev_recon_delta
            
        return tokens

    def reconstruct_trajectory(self, start_pos, tokens, initial_delta=None):
        curr_pos = np.array(start_pos, dtype=np.float32)
        trajectory = [curr_pos.copy()]
        prev_delta = np.array(initial_delta if initial_delta is not None else [0.0, 0.0], dtype=np.float32)
        
        for token in tokens:
            dd = np.array(self.decode_token(token))
            curr_delta = prev_delta + dd
            curr_pos += curr_delta
            trajectory.append(curr_pos.copy())
            prev_delta = curr_delta
            
        return np.array(trajectory)
