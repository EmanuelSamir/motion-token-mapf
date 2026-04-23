import numpy as np
import torch
import os
from src.data.dataset import TrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

def run_sensitivity_test():
    data_dir = "data/trajectories_clean"
    # Load dataset (already cleaned with episode detection)
    dataset = TrajectoryDataset(data_dir, num_agents=1, sample_ratio=0.05)
    
    num_bins_list = [13, 25]
    max_dd_list = [0.2, 0.5, 1.0, 1.5, 2.0]
    
    print(f"{'Bins':<6} | {'MaxDD':<6} | {'Mean Err (m)':<15} | {'Max Err (m)':<15} | {'Clipping %':<12}")
    print("-" * 65)

    for nb in num_bins_list:
        for mdd in max_dd_list:
            tokenizer = MotionTokenizer(num_bins=nb, max_delta_delta=mdd)
            
            errors = []
            max_errors = []
            clips = 0
            total_steps = 0
            
            for i in range(len(dataset)):
                sample = dataset[i]
                gt = sample['gt_future'].numpy()[0] # [21, 2]
                init_d = sample['initial_deltas'].numpy()[0]
                
                # Check for clipping
                deltas = np.diff(gt, axis=0) # [20, 2]
                prev_d = init_d
                for d in deltas:
                    dd = d - prev_d
                    if np.any(np.abs(dd) > mdd):
                        clips += 1
                    total_steps += 1
                    prev_d = d
                
                # Tokenize and Reconstruct
                tokens = tokenizer.tokenize_trajectory(gt, initial_delta=init_d)
                recon = tokenizer.reconstruct_trajectory(gt[0], tokens, initial_delta=init_d)
                
                # L2 distance at each step
                dist = np.linalg.norm(gt - recon, axis=1)
                errors.extend(dist.tolist())
                max_errors.append(np.max(dist))
            
            print(f"{nb:<6} | {mdd:<6} | {np.mean(errors):<15.4f} | {np.max(max_errors):<15.4f} | {100*clips/total_steps:<12.1f}")

if __name__ == "__main__":
    run_sensitivity_test()
