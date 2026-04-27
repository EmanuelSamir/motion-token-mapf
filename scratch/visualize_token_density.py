import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data.hf_dataset import HFTrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

def analyze_vocabulary_fit(path, num_samples=1000):
    dataset = HFTrajectoryDataset(path)
    tokenizer = MotionTokenizer()
    
    token_counts = np.zeros((tokenizer.num_bins, tokenizer.num_bins))
    clipping_events = 0
    total_steps = 0
    
    print(f"Analyzing token density and physical fit on {num_samples} samples...")
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]
        
        # We RE-TOKENIZE here to detect clipping (instead of reading stored tokens)
        gt_futures = sample['gt_future'].numpy()
        masks = sample['m_mask'].numpy()
        init_deltas = sample['initial_deltas'].numpy()
        
        for n in range(gt_futures.shape[0]):
            if masks[n].sum() < 3: continue
            
            traj = gt_futures[n][masks[n] > 0]
            idelta = init_deltas[n]
            
            # Manual step-by-step to detect clipping
            prev_v_bin = tokenizer._get_velocity_bin(idelta)
            curr_pos = traj[0]
            
            ox, oy = np.meshgrid(tokenizer.offsets, tokenizer.offsets, indexing='ij')
            token_offsets = np.stack([ox.flatten(), oy.flatten()], axis=1)
            
            for t in range(1, len(traj)):
                target_pos = traj[t]
                total_steps += 1
                
                # Check if ideal bin is outside limits
                ideal_v = target_pos - curr_pos
                ideal_bin = tokenizer._get_velocity_bin(ideal_v)
                
                # If the jump needed is > offset_range (6), it will be clipped
                needed_offset = ideal_bin - prev_v_bin
                if np.any(np.abs(needed_offset) > tokenizer.offset_range):
                    clipping_events += 1
                
                # Standard greedy step (with clipping)
                candidate_bins = np.clip(prev_v_bin + token_offsets, 0, tokenizer.v_bins_count - 1)
                cv_x = tokenizer.v_grid_x[candidate_bins[:, 0]]
                cv_y = tokenizer.v_grid_y[candidate_bins[:, 1]]
                candidate_v = np.stack([cv_x, cv_y], axis=1)
                
                candidate_pos = curr_pos + candidate_v
                errors = np.sum((candidate_pos - target_pos)**2, axis=1)
                best_token = np.argmin(errors)
                token_counts[best_token // 13, best_token % 13] += 1
                
                prev_v_bin = candidate_bins[best_token]
                curr_pos = curr_pos + np.array([tokenizer.v_grid_x[prev_v_bin[0]], tokenizer.v_grid_y[prev_v_bin[1]]])

    # Visualization
    log_density = np.log10(token_counts + 1)
    log_density = log_density / (log_density.max() + 1e-6)
    
    plt.figure(figsize=(8, 7))
    plt.imshow(log_density, cmap='magma', origin='lower')
    plt.colorbar(label='Log10 Density')
    plt.title(f"Corrected Axes Density (X=Speed, Y=Lane)\nClipping: {(clipping_events/total_steps)*100:.2f}% of steps")
    plt.xlabel("Y-Offset (Lane)")
    plt.ylabel("X-Offset (Speed)")
    
    save_path = "scratch/corrected_token_density.png"
    plt.savefig(save_path)
    print(f"\n--- Vocabulary Analysis ---")
    print(f"Total steps analyzed: {total_steps}")
    print(f"Clipping frequency: {(clipping_events/total_steps)*100:.4f}%")
    print(f"Conclusion: {'READY' if (clipping_events/total_steps) < 0.01 else 'NEEDS ADJUSTMENT'}")
    print(f"Map saved to {save_path}")

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    analyze_vocabulary_fit(PATH)
