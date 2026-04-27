import os
import sys
sys.path.append(os.getcwd())
from src.data.hf_dataset import HFTrajectoryDataset
import torch

def debug_coordinates():
    base_data_path = "data/trajectories"
    runs = [os.path.join(base_data_path, d) for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))]
    if not runs:
        print("No data found")
        return
    
    dataset = HFTrajectoryDataset(runs[0])
    sample = dataset[0]
    
    history = sample['history'] # [N, T, 5]
    gt_future = sample['gt_future'] # [N, T_pred+1, 2]
    
    print(f"--- Full Feature Debug ---")
    print(f"Features are: [x, y, vx, vy, yaw]")
    for n in range(min(2, history.shape[0])):
        print(f"\nAGENT {n}:")
        print(f"  Pos at t=0: {history[n, -1, :2].numpy()}")
        print(f"  Vel at t=0: {history[n, -1, 2:4].numpy()}")
        print(f"  Yaw at t=0: {history[n, -1, 4].item():.4f}")
        
    print("\nRELATIONSHIP:")
    dist = torch.norm(history[0, -1, :2] - history[1, -1, :2])
    print(f"Distance between Agent 0 and Agent 1 at t=0: {dist.item():.2f}m")
    
    # Check if Agent 1 is also not centered
    if torch.abs(history[1, -1, :2]).max() > 1.0:
        print("\n🚩 CONFIRMED: All agents share a 'Scene' coordinate system.")
        print("Recommendation: Normalize to Agent-Centric (0,0, north) for each agent.")

if __name__ == "__main__":
    debug_coordinates()
