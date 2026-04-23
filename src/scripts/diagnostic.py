import numpy as np
import torch
import os
from src.data.dataset import TrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

def run_diagnostic():
    data_dir = "data/trajectories"
    dataset = TrajectoryDataset(data_dir, num_agents=1, sample_ratio=0.1)
    
    tokenizer = MotionTokenizer(num_bins=13, max_delta_delta=2.0)
    
    anomalies = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        gt = sample['gt_future'].numpy()[0]
        init_d = sample['initial_deltas'].numpy()[0]
        
        # Tokenize and Reconstruct
        tokens = tokenizer.tokenize_trajectory(gt, initial_delta=init_d)
        recon = tokenizer.reconstruct_trajectory(gt[0], tokens, initial_delta=init_d)
        
        err = np.linalg.norm(gt[-1] - recon[-1])
        if err > 10.0:
            anomalies += 1
            if anomalies <= 5:
                # Debug this sample
                deltas = np.diff(gt, axis=0) # [20, 2]
                all_d = np.vstack([init_d, deltas]) # [21, 2]
                dd = np.diff(all_d, axis=0)
                
                print(f"\n--- ANOMALY {i} ---")
                print(f"Max recon error: {err:.2f}m")
                print(f"Initial Delta: {init_d}")
                print(f"First Future Delta: {deltas[0]}")
                print(f"First DD: {dd[0]}")
                print(f"Max Abs DD: {np.max(np.abs(dd))}")
                print(f"Max Abs Speed: {np.max(np.abs(all_d))/0.2}")
                # print(f"GT Traj:\n{gt[:5]}")
    
    print(f"\nTotal anomalies (>10m error): {anomalies} / {len(dataset)}")

if __name__ == "__main__":
    run_diagnostic()
