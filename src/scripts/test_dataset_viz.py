import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data.dataset import TrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

def test_dataset_visualization():
    # Update to the newly collected dataset directory
    data_dir = "data/trajectories/hdv_1000_ep_abs_coords"
    if not os.path.exists(data_dir):
        # Fallback to base dir if specific tag folder not found
        data_dir = "data/trajectories"
        print(f"Target directory not found, falling back to {data_dir}")

    tokenizer = MotionTokenizer(num_bins=13, max_delta_delta=1.0)
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        history_len=10,
        prediction_len=25,
        tokenizer=tokenizer,
        sample_ratio=1.0,
        stride=10
    )

    if len(dataset) == 0:
        print("No samples found. Check if data collection was successful.")
        return

    os.makedirs("test_viz", exist_ok=True)
    
    # Process all samples
    num_samples = len(dataset)
    print(f"📊 Total samples to process: {num_samples}")
    
    for idx in range(num_samples):
        # Create subfolders every 100 samples to keep it organized
        sub_folder = f"test_viz/batch_{idx // 100}"
        os.makedirs(sub_folder, exist_ok=True)
        
        sample = dataset[idx]
        history = sample['history'].numpy() # [MaxN, 10, 5]
        gt_future = sample['gt_future'].numpy() # [MaxN, 26, 2]
        tokens = sample['tokens'].numpy()
        init_deltas = sample['initial_deltas'].numpy() # [MaxN, 2]
        m_mask = sample['m_mask'].numpy() # [MaxN, 26]
        num_agents = int(sample['num_agents'].item())
        
        MaxN = dataset.max_agents_per_scene
        T_f = dataset.prediction_len
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Color palette for agents
        colors = plt.cm.get_cmap('tab10', MaxN)
        
        for n in range(num_agents):
            color = colors(n)
            
            # 1. Plot Real History (10 points)
            h_x = history[n, :, 0]
            h_y = history[n, :, 1]
            ax.plot(h_x, h_y, color=color, marker='.', markersize=8, 
                    alpha=0.3, linestyle=':', label=f'Agent {n} Hist' if n==0 else "")
            
            # 2. Plot REAL Ground Truth Future
            active_future_mask = m_mask[n] > 0
            f_x = gt_future[n, active_future_mask, 0]
            f_y = gt_future[n, active_future_mask, 1]
            
            ax.plot(f_x, f_y, color=color, marker='o', markersize=5, 
                    linestyle='-', linewidth=2, label=f'Agent {n} GT' if n==0 else "")
            
            # 3. Plot Token Reconstruction
            agent_tokens = []
            for t in range(T_f):
                agent_tokens.append(tokens[t * MaxN + n])
            
            valid_tokens = [tok for tok in agent_tokens if tok != 170]
            if valid_tokens:
                start_pos = history[n, -1, :2]
                traj_recon = tokenizer.reconstruct_trajectory(start_pos, valid_tokens, initial_delta=init_deltas[n])
                
                ax.plot(traj_recon[:, 0], traj_recon[:, 1], color='black', marker='x', markersize=4, 
                        linestyle='--', alpha=0.6, label='Token Recon' if n==0 else "")

        # Visualization improvements
        ax.set_title(f"Dataset Inspection | Sample {idx} | Absolute Coordinates ({num_agents} Agents)")
        ax.set_xlabel("Global X (meters)")
        ax.set_ylabel("Global Y (meters)")
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # Force a reasonable aspect ratio for coordinates
        ax.set_aspect('equal', adjustable='box')
        ax.axhline(y=10.0, color='gray', linestyle='--', alpha=0.2)

        plt.tight_layout()
        save_path = os.path.join(sub_folder, f"sample_{idx:05d}.png")
        plt.savefig(save_path)
        plt.close()
        
        if idx % 10 == 0:
            print(f"⌛ Progress: {idx}/{num_samples} ({(idx/num_samples)*100:.1f}%)")

    print(f"✨ Finished. All visualizations saved in test_viz/")

if __name__ == "__main__":
    test_dataset_visualization()
