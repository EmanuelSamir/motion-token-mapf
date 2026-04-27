import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.tokenizer import MotionTokenizer
from src.data.hf_dataset import HFTrajectoryDataset

def test_tokenization_reconstruction(path, num_samples=500):
    dataset = HFTrajectoryDataset(path)
    tokenizer = MotionTokenizer()
    
    os.makedirs("images", exist_ok=True)
    
    total_recon_error = 0
    total_points = 0
    max_errors = []
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    print(f"Testing tokenization reconstruction on {len(indices)} random samples from {path}...")
    
    for i in tqdm(indices):
        sample = dataset[int(i)]
        
        gt_futures = sample['gt_future'].numpy()
        masks = sample['m_mask'].numpy()
        init_deltas = sample['initial_deltas'].numpy()
        N_max = gt_futures.shape[0]
        
        for n in range(N_max):
            if masks[n].sum() < 2:
                continue
                
            start_pos = gt_futures[n][0]
            idelta = init_deltas[n]
            traj_gt = gt_futures[n][masks[n] > 0]
            
            # 1. Tokenize GT with the NEW tokenizer
            new_tokens = tokenizer.tokenize_trajectory(traj_gt, initial_delta=idelta)
            
            # 2. Reconstruct from these tokens
            traj_recon = tokenizer.reconstruct_trajectory(start_pos, new_tokens, initial_delta=idelta)
            
            # 3. Compare
            L = min(len(traj_gt), len(traj_recon))
            diff = traj_gt[:L] - traj_recon[:L]
            rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            max_err = np.max(np.sqrt(np.sum(diff**2, axis=1)))
            
            total_recon_error += np.sum(np.sqrt(np.sum(diff**2, axis=1)))
            total_points += L
            max_errors.append(max_err)

            if max_err > 0.1:
                # Plot errors for visual check
                plt.figure(figsize=(10, 6))
                plt.plot(traj_gt[:, 0], traj_gt[:, 1], 'g-', label='Ground Truth', alpha=0.5, marker='o', markersize=3)
                plt.plot(traj_recon[:, 0], traj_recon[:, 1], 'r--', label='Reconstructed', alpha=0.8, marker='x', markersize=4)
                plt.title(f"Sample {i} Agent {n} | RMSE: {rmse:.4f} | Max Err: {max_err:.4f}")
                plt.legend()
                plt.axis("equal")
                plt.grid(True, alpha=0.3)
                plt.savefig(f"images/ultim_recon_s{i}_a{n}.png")
                plt.close()

    print(f"\n--- Reconstruction Performance ---")
    print(f"Average Reconstruction Error: {total_recon_error / (total_points+1e-6):.6f}m")
    if max_errors:
        print(f"Max Error observed: {max(max_errors):.4f}m")
        print(f"95th percentile Max Error: {np.percentile(max_errors, 95):.4f}m")
    
    if (total_recon_error / (total_points+1e-6)) < 0.1:
        print("✅ SUCCESS: Tokenizer reconstruction is accurate.")
    else:
        print("❌ WARNING: High reconstruction error detected!")

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    test_tokenization_reconstruction(PATH, num_samples=500)
