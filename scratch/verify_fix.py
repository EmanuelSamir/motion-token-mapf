import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.tokenizer import MotionTokenizer
from src.data.hf_dataset import HFTrajectoryDataset

def debug_sample(path, sample_idx=265, agent_idx=3):
    dataset = HFTrajectoryDataset(path)
    tokenizer = MotionTokenizer()
    
    os.makedirs("images/debug_fix", exist_ok=True)
    
    sample = dataset[sample_idx]
    gt_futures = sample['gt_future'].numpy()
    masks = sample['m_mask'].numpy()
    init_deltas = sample['initial_deltas'].numpy()
    
    traj_gt = gt_futures[agent_idx][masks[agent_idx] > 0]
    idelta = init_deltas[agent_idx]
    
    print(f"Agent {agent_idx} | New Initial Delta: {idelta}")
    
    # Regenerate tokens from GT
    tokens = tokenizer.tokenize_trajectory(traj_gt, initial_delta=idelta)
    traj_recon = tokenizer.reconstruct_trajectory(traj_gt[0], tokens, initial_delta=idelta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(traj_gt[:, 0], traj_gt[:, 1], 'g-', label='Ground Truth', marker='o', alpha=0.3)
    plt.plot(traj_recon[:, 0], traj_recon[:, 1], 'r--', label='Reconstructed (Fixed idelta)', marker='x', alpha=0.8)
    plt.title(f"Sample {sample_idx} Agent {agent_idx} | Fixed idelta: {idelta}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.2)
    plt.savefig(f"images/debug_fix/sample_{sample_idx}_agent_{agent_idx}.png")
    print(f"Plot saved to images/debug_fix/sample_{sample_idx}_agent_{agent_idx}.png")

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    debug_sample(PATH)
