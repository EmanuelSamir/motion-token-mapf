import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import glob
from tqdm import tqdm
from src.data.hf_dataset import HFTrajectoryDataset
from src.models.motion_lm_module import MotionLMLightningModule
from omegaconf import OmegaConf

def find_latest_checkpoint():
    search_paths = [
        os.path.join("checkpoints", "**", "*.ckpt"),
        os.path.join("lightning_logs", "version_*", "checkpoints", "*.ckpt"),
    ]
    checkpoints = []
    for path in search_paths:
        checkpoints.extend(glob.glob(path, recursive=True))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def evaluate_rollout(num_samples=100):
    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print("❌ No checkpoint found. Please train the model first.")
        return

    print(f"🧐 Loading checkpoint: {ckpt_path}")
    
    # Load model (Assuming standard config names)
    # We can try to load config from the checkpoint later if needed
    model = MotionLMLightningModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset_path = 'data/trajectories/run_20260423_015507' # Using your standard dataset
    dataset = HFTrajectoryDataset(dataset_path)
    
    ade_list = []
    fde_list = []
    
    print(f"🎬 Evaluating {num_samples} samples with Autoregressive Rollout...")
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        batch = dataset[i]
        # Wrap in batch dimension
        history = batch['history'].unsqueeze(0).to(model.device)
        gt_future = batch['gt_future'].numpy()
        m_mask = batch['m_mask'].numpy()
        init_deltas = batch['initial_deltas'].numpy()
        
        # Rollout
        with torch.no_grad():
            rollout_tokens = model.autoregressive_rollout(history)
            tokens_np = rollout_tokens[0].cpu().numpy()
            
        N = history.shape[1]
        T_steps = model.hparams.model_config["max_timesteps"]
        
        for n in range(N):
            if m_mask[n].sum() < 2: continue
            
            agent_tokens = [tokens_np[tt * N + n] for tt in range(T_steps)]
            
            traj_pred = model.tokenizer.reconstruct_trajectory(
                gt_future[n, 0], agent_tokens, initial_delta=init_deltas[n]
            )
            
            traj_gt = gt_future[n][m_mask[n] > 0]
            min_len = min(len(traj_pred), len(traj_gt))
            
            if min_len < 2: continue
            
            dist = np.linalg.norm(traj_pred[:min_len] - traj_gt[:min_len], axis=1)
            ade_list.append(np.mean(dist))
            fde_list.append(dist[-1])

    print(f"\n--- Rollout Evaluation Results ({len(ade_list)} non-empty agents) ---")
    print(f"Final ADE: {np.mean(ade_list):.4f}m")
    print(f"Final FDE: {np.mean(fde_list):.4f}m")

if __name__ == "__main__":
    evaluate_rollout(num_samples=100)
