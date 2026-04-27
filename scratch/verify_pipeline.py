import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from src.data.hf_dataset import HFTrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

def verify_end_to_end():
    dataset = HFTrajectoryDataset('data/trajectories/run_20260423_015507')
    tokenizer = MotionTokenizer()
    
    # Get a batch
    sample = dataset[0]
    tokens = sample['tokens']        # [N*T] Interleaved
    agent_ids = sample['agent_ids']  # [N*T]
    time_ids = sample['time_ids']    # [N*T]
    gt_future = sample['gt_future']  # [N, T+1, 2]
    m_mask = sample['m_mask']        # [N, T+1]
    ideltas = sample['initial_deltas']
    
    N = sample['num_agents'].item()
    T = int(len(tokens) / N)
    
    print(f"--- Pipeline Verification ---")
    print(f"Batch has {N} agents and {T} timesteps.")
    
    errors = []
    
    for n in range(N):
        if m_mask[n].sum() < 5: continue
        
        # 1. Extract tokens for this agent from the interleaved stream
        # This simulates exactly what the tokenizer.reconstruct needs
        agent_tokens = []
        for t in range(T):
            # Find index in interleaved where time=t and agent=n
            idx = t * N + n
            # Triple check IDs
            assert agent_ids[idx] == n, f"Identity mismatch at t={t}, expected {n} got {agent_ids[idx]}"
            assert time_ids[idx] == t, f"Time mismatch at t={t}, expected {t} got {time_ids[idx]}"
            agent_tokens.append(tokens[idx].item())
            
        # 2. Reconstruct
        start_pos = gt_future[n, 0].numpy()
        idelta = ideltas[n].numpy()
        
        traj_recon = tokenizer.reconstruct_trajectory(start_pos, agent_tokens, initial_delta=idelta)
        
        # 3. Compare with original GT in the batch
        active_mask = m_mask[n] > 0
        traj_gt = gt_future[n][active_mask].numpy()
        
        min_len = min(len(traj_recon), len(traj_gt))
        rmse = np.sqrt(np.mean(np.sum((traj_recon[:min_len] - traj_gt[:min_len])**2, axis=1)))
        errors.append(rmse)
        print(f"Agent {n} Reconstruction RMSE: {rmse:.6f}m")

    avg_error = np.mean(errors)
    print(f"\nFinal Verdict:")
    if avg_error < 0.15:
        print(f"✅ PIPELINE VERIFIED. Average error {avg_error*100:.2f}cm is perfect.")
    else:
        print(f"❌ PIPELINE ERROR. Error {avg_error:.4f}m is too high. Investigating...")

if __name__ == "__main__":
    verify_end_to_end()
