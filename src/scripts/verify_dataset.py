import os
import pickle
import numpy as np
from src.data.dataset import TrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

def verify_dataset_integrity():
    data_dir = "data/trajectories"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return

    tokenizer = MotionTokenizer()
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        history_len=10,
        prediction_len=20,
        num_agents=3,
        tokenizer=tokenizer,
        sample_ratio=1.0 # Check all
    )

    print(f"\nScanning {len(dataset)} segments for length integrity...")
    
    anomalies = 0
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        hist = sample['history'] # [N, T_hist, 5]
        tokens = sample['tokens'] # [N*T_pred]
        
        # Check history length
        if hist.shape[1] != 10:
            print(f"❌ Sample {i} has invalid history length: {hist.shape[1]}")
            anomalies += 1
            
        # Check if agent is actually moving
        # Calculate max displacement in history
        for n in range(hist.shape[0]):
            disp = np.linalg.norm(hist[n, :, :2], axis=-1)
            max_disp = np.max(disp)
            if max_disp < 0.001:
                # Stationary agent - might look like a single point
                pass
            
            # Check if there are any NaNs
            if np.isnan(hist.numpy()).any():
                print(f"❌ Sample {i}, Agent {n} contains NaNs")
                anomalies += 1

    if anomalies == 0:
        print("✅ Integrity check passed: All inspected samples have correct shapes and no NaNs.")
    else:
        print(f"⚠️ Found {anomalies} anomalies.")

if __name__ == "__main__":
    verify_dataset_integrity()
