import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from src.utils.tokenizer import MotionTokenizer
from src.data.hf_dataset import HFTrajectoryDataset

def test_and_viz(path, num_samples=10):
    dataset = HFTrajectoryDataset(path)
    tokenizer = MotionTokenizer()
    
    os.makedirs("images/test_tokenization", exist_ok=True)
    
    print(f"Visualizing tokenization on {num_samples} samples...")
    
    for i in range(num_samples):
        sample = dataset[i]
        gt_futures = sample['gt_future'].numpy()
        masks = sample['m_mask'].numpy()
        init_deltas = sample['initial_deltas'].numpy()
        tokens_interleaved = sample['tokens'].numpy()
        
        N_max = gt_futures.shape[0]
        T_pred = gt_futures.shape[1] - 1
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for n in range(min(5, N_max)): # Plot up to 5 agents
            if masks[n].sum() < 2:
                continue
                
            traj_gt = gt_futures[n][masks[n] > 0]
            idelta = init_deltas[n]
            
            # 1. Regenerate tokens from GT
            new_tokens = tokenizer.tokenize_trajectory(traj_gt, initial_delta=idelta)
            traj_recon_new = tokenizer.reconstruct_trajectory(traj_gt[0], new_tokens, initial_delta=idelta)
            
            # 2. Extract tokens from dataset
            stored_tokens = [tokens_interleaved[t * N_max + n] for t in range(T_pred)]
            valid_stored_tokens = [t for t in stored_tokens if t < 169]
            traj_recon_stored = tokenizer.reconstruct_trajectory(traj_gt[0], valid_stored_tokens, initial_delta=idelta)
            
            # Plot on axes[0]: Regenerated vs GT
            axes[0].plot(traj_gt[:, 0], traj_gt[:, 1], 'g-', alpha=0.3)
            axes[0].scatter(traj_gt[:, 0], traj_gt[:, 1], c='g', s=10)
            axes[0].plot(traj_recon_new[:, 0], traj_recon_new[:, 1], 'r--', alpha=0.6)
            axes[0].scatter(traj_recon_new[:, 0], traj_recon_new[:, 1], c='r', marker='x', s=15)
            
            # Plot on axes[1]: Stored vs GT
            axes[1].plot(traj_gt[:, 0], traj_gt[:, 1], 'g-', alpha=0.3)
            axes[1].scatter(traj_gt[:, 0], traj_gt[:, 1], c='g', s=10)
            axes[1].plot(traj_recon_stored[:, 0], traj_recon_stored[:, 1], 'b--', alpha=0.6)
            axes[1].scatter(traj_recon_stored[:, 0], traj_recon_stored[:, 1], c='b', marker='+', s=15)
            
        axes[0].set_title("Regenerated Tokens vs GT")
        axes[1].set_title("Stored Tokens vs GT")
        for ax in axes:
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
        plt.tight_layout()
        plt.savefig(f"images/test_tokenization/sample_{i}.png")
        plt.close()

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    test_and_viz(PATH, num_samples=20)
