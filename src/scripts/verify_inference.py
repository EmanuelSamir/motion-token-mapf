import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.inference import MotionLMInferenceEngine
from src.data.hf_dataset import HFTrajectoryDataset
from src.utils.tokenizer import MotionTokenizer
import os

def verify_inference(checkpoint_path, hf_dataset_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = MotionLMInferenceEngine(checkpoint_path, device=device)
    dataset = HFTrajectoryDataset(hf_dataset_path)
    tokenizer = MotionTokenizer()
    
    # Take a sample
    sample = dataset[0]
    history = sample['history'] # [N, 10, 5]
    gt_tokens = sample['tokens'] # [N*T]
    initial_deltas = sample['initial_deltas'] # [N, 2]
    num_agents = sample['num_agents']
    
    print(f"Sample with {num_agents} agents.")
    print(f"History (Agent 0, first 2 steps):\n{history[0, :2]}")
    print(f"Initial Delta (Agent 0): {initial_deltas[0]}")
    
    # Check GT Tokens for Agent 0
    agent0_gt_tokens = [gt_tokens[t*10 + 0].item() for t in range(20)]
    print(f"GT Tokens (Agent 0): {agent0_gt_tokens[:5]}...")
    for tok in agent0_gt_tokens[:3]:
        dd = tokenizer.decode_token(tok)
        print(f"  Token {tok} -> dd: {dd}")
    # Prediction
    # predict_joint_v2 now uses SOS=84 internally
    pred_dict = engine.predict_joint_v2(history, max_timesteps=20)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in range(num_agents):
        # 1. Ground Truth (from GT tokens)
        # Interleaved [A1_T1, A2_T1, ..., A1_T2, ...]
        agent_gt_tokens = [gt_tokens[t*10 + n].item() for t in range(20)]
        start_pos = history[n, -1, :2].numpy()
        
        # NOTE: For reconstruction logic, the initial_delta remains physical.
        # It's only the model input during the first autoregressive step that is SOS.
        traj_gt = tokenizer.reconstruct_trajectory(start_pos, agent_gt_tokens, initial_delta=initial_deltas[n].numpy())
        
        # 2. Prediction
        agent_pred_tokens = pred_dict[n]
        traj_pred = tokenizer.reconstruct_trajectory(start_pos, agent_pred_tokens, initial_delta=initial_deltas[n].numpy())
        
        ax.plot(traj_gt[:, 0], traj_gt[:, 1], 'g-', alpha=0.5, label=f'Agent {n} GT' if n==0 else "")
        ax.plot(traj_pred[:, 0], traj_pred[:, 1], 'r--', label=f'Agent {n} Pred' if n==0 else "")
        
        print(f"Agent {n} Pred X drift: {traj_pred[-1, 0] - traj_pred[0, 0]:.2f}m")
        print(f"Agent {n} GT X drift: {traj_gt[-1, 0] - traj_gt[0, 0]:.2f}m")

    ax.set_title("Inference Verification (HF Sample)")
    ax.legend()
    plt.savefig("inference_verification.png")
    print("Verification plot saved to inference_verification.png")

if __name__ == "__main__":
    CKPT = "checkpoints/motion-lm-epoch=35-val/loss=1.5860.ckpt"
    HF_DATA = "data/trajectories/hdv_1000_ep_hf"
    
    if os.path.exists(CKPT) and os.path.exists(HF_DATA):
        verify_inference(CKPT, HF_DATA)
    else:
        print(f"Missing CKPT or HF_DATA: {CKPT} {os.path.exists(CKPT)}, {HF_DATA} {os.path.exists(HF_DATA)}")
