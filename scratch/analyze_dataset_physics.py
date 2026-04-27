import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from tqdm import tqdm
from src.data.hf_dataset import HFTrajectoryDataset

def analyze_physics(path, num_samples=500):
    dataset = HFTrajectoryDataset(path)
    print(f"Analyzing physics on {num_samples} samples...")
    
    all_deltas = []
    all_delta_deltas = [] # Accelerations
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]
        # gt_future is [N, T+1, 2]
        gt = sample['gt_future'].numpy()
        mask = sample['m_mask'].numpy()
        
        for n in range(gt.shape[0]):
            if mask[n].sum() < 3: continue
            
            traj = gt[n][mask[n] > 0]
            
            # Deltas (Velocity-ish)
            deltas = np.diff(traj, axis=0) # [T, 2]
            all_deltas.extend(deltas.flatten())
            
            # Delta-Deltas (Acceleration-ish)
            if len(deltas) > 1:
                # In Verlet: dd = delta[t] - delta[t-1]
                dds = np.diff(deltas, axis=0) # [T-1, 2]
                all_delta_deltas.extend(dds.tolist())
                
    all_deltas = np.array(all_deltas)
    all_delta_deltas = np.array(all_delta_deltas)
    
    print("\n--- Velocity (Deltas) Distribution ---")
    print(f"Mean: {np.mean(all_deltas):.4f}m")
    print(f"Std:  {np.std(all_deltas):.4f}m")
    print(f"Min:  {np.min(all_deltas):.4f}m")
    print(f"Max:  {np.max(all_deltas):.4f}m")
    print(f"99th Percentile (abs): {np.percentile(np.abs(all_deltas), 99):.4f}m")

    all_dds_x = np.array(all_delta_deltas)[:, 0]
    all_dds_y = np.array(all_delta_deltas)[:, 1]
    
    print("\n--- Acceleration X (Delta-Deltas) ---")
    print(f"Min: {np.min(all_dds_x):.4f}m | Max: {np.max(all_dds_x):.4f}m")
    print(f"99th Percentile (abs): {np.percentile(np.abs(all_dds_x), 99):.4f}m")

    print("\n--- Acceleration Y (Delta-Deltas) ---")
    print(f"Min: {np.min(all_dds_y):.4f}m | Max: {np.max(all_dds_y):.4f}m")
    print(f"99th Percentile (abs): {np.percentile(np.abs(all_dds_y), 99):.4f}m")
    
    # Check if 1.0 is enough
    coverage = np.mean(np.abs(all_delta_deltas) <= 1.0) * 100
    print(f"\nCoverage with current max_delta_delta = 1.0: {coverage:.2f}%")

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    analyze_physics(PATH)
