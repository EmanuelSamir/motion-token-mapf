import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data.hf_dataset import HFTrajectoryDataset

def analyze_motion(path, num_samples=500):
    dataset = HFTrajectoryDataset(path)
    
    os.makedirs("images", exist_ok=True)
    
    all_vx = []
    all_vy = []
    all_ax = []
    all_ay = []
    
    print(f"Analyzing motion on {num_samples} samples from {path}...")
    
    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]
        gt_futures = sample['gt_future'].numpy()
        masks = sample['m_mask'].numpy()
        init_deltas = sample['initial_deltas'].numpy()
        N_max = gt_futures.shape[0]
        
        for n in range(N_max):
            if masks[n].sum() < 3: # Need at least 3 points for acceleration
                continue
            
            traj_gt = gt_futures[n][masks[n] > 0]
            
            # Velocities (displacements per step)
            # v_t = p_t - p_{t-1}
            # We also have the initial velocity (init_deltas)
            
            v = np.diff(traj_gt, axis=0) # [T-1, 2]
            
            # Prepend initial delta to have the sequence of velocities
            v_seq = np.vstack([init_deltas[n], v]) # [T, 2]
            
            all_vx.extend(v_seq[:, 0])
            all_vy.extend(v_seq[:, 1])
            
            # Accelerations (change in velocity)
            # a_t = v_t - v_{t-1}
            a = np.diff(v_seq, axis=0)
            
            all_ax.extend(a[:, 0])
            all_ay.extend(a[:, 1])

    dt = 0.2
    all_vx = np.array(all_vx) / dt
    all_vy = np.array(all_vy) / dt
    all_ax = np.array(all_ax) / (dt**2)
    all_ay = np.array(all_ay) / (dt**2)
    
    print(f"\n--- Physical Statistics (dt={dt}s) ---")
    print(f"Velocity X [m/s]: min={all_vx.min():.4f}, max={all_vx.max():.4f}, mean={all_vx.mean():.4f}, std={all_vx.std():.4f}")
    print(f"Velocity Y [m/s]: min={all_vy.min():.4f}, max={all_vy.max():.4f}, mean={all_vy.mean():.4f}, std={all_vy.std():.4f}")
    print(f"Accel X [m/s^2]:    min={all_ax.min():.4f}, max={all_ax.max():.4f}, mean={all_ax.mean():.4f}, std={all_ax.std():.4f}")
    print(f"Accel Y [m/s^2]:    min={all_ay.min():.4f}, max={all_ay.max():.4f}, mean={all_ay.mean():.4f}, std={all_ay.std():.4f}")

    print(f"\n--- Percentiles (Accel X) ---")
    for p in [0.1, 1, 5, 50, 95, 99, 99.9]:
        print(f"  {p}%: {np.percentile(all_ax, p):.4f}")
    print(f"\n--- Percentiles (Accel Y) ---")
    for p in [0.1, 1, 5, 50, 95, 99, 99.9]:
        print(f"  {p}%: {np.percentile(all_ay, p):.4f}")

    # Proposed new limits (in m/s)
    NEW_MAX_V_X_MS = 25.0 
    NEW_MAX_V_Y_MS = 10.0 
    V_BINS = 128
    
    v_step_x_ms = (2 * NEW_MAX_V_X_MS) / (V_BINS - 1)
    v_step_y_ms = (2 * NEW_MAX_V_Y_MS) / (V_BINS - 1)
    
    accel_step_x = v_step_x_ms / dt
    accel_step_y = v_step_y_ms / dt
    
    print(f"\n--- Asymmetric Offset Analysis ---")
    # Candidate X: [0, 1, 2, 4, -1, -2, -5, -10, -20]
    cand_x = np.array([0, 1, -1, 2, -2, 5, -5, -12, -25]) # 9 tokens
    # Candidate Y: [0, 1, -1, 4, -4, -15, -40]
    cand_y = np.array([0, 1, -1, 4, -4, 10, -20]) # 7 tokens
    
    def check_coverage(data, cand, step):
        # For each point in data, find if there is an offset in cand that covers it
        # Real coverage in greedy search is "is the closest offset better than nothing"
        # But here let's just see if we can represent it with < 1.0 step error
        covered = 0
        for val in data:
            best_diff = np.min(np.abs(val - cand * step))
            if best_diff < step / 2:
                covered += 1
        return covered / len(data) * 100

    print(f"Cand X {cand_x} coverage: {check_coverage(all_ax, cand_x, accel_step_x):.2f}%")
    print(f"Cand Y {cand_y} coverage: {check_coverage(all_ay, cand_y, accel_step_y):.2f}%")
    
    # 2D Heatmaps (Joint Distribution)
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    
    # Velocity 2D (Linear)
    h_v = axs[0, 0].hist2d(all_vx, all_vy, bins=100, cmap='viridis', cmin=1)
    axs[0, 0].set_title("Velocity 2D Distribution (Linear)")
    axs[0, 0].set_xlabel("Vx [m/s]")
    axs[0, 0].set_ylabel("Vy [m/s]")
    plt.colorbar(h_v[3], ax=axs[0, 0])
    
    # Velocity 2D (Log)
    import matplotlib.colors as mcolors
    h_v_log = axs[0, 1].hist2d(all_vx, all_vy, bins=100, cmap='viridis', cmin=1, norm=mcolors.LogNorm())
    axs[0, 1].set_title("Velocity 2D Distribution (Log)")
    axs[0, 1].set_xlabel("Vx [m/s]")
    axs[0, 1].set_ylabel("Vy [m/s]")
    plt.colorbar(h_v_log[3], ax=axs[0, 1])
    
    # Acceleration 2D (Linear) - Focused
    h_a = axs[1, 0].hist2d(all_ax, all_ay, bins=100, range=[[-15, 5], [-20, 10]], cmap='magma', cmin=1)
    axs[1, 0].set_title("Acceleration 2D Distribution (Linear, Focused)")
    axs[1, 0].set_xlabel("Ax [m/s^2]")
    axs[1, 0].set_ylabel("Ay [m/s^2]")
    plt.colorbar(h_a[3], ax=axs[1, 0])
    
    # Acceleration 2D (Log) - Full range
    h_a_log = axs[1, 1].hist2d(all_ax, all_ay, bins=100, cmap='magma', cmin=1, norm=mcolors.LogNorm())
    axs[1, 1].set_title("Acceleration 2D Distribution (Log, Full Range)")
    axs[1, 1].set_xlabel("Ax [m/s^2]")
    axs[1, 1].set_ylabel("Ay [m/s^2]")
    plt.colorbar(h_a_log[3], ax=axs[1, 1])
    
    plt.tight_layout()
    plt.savefig("images/motion_2d_analysis.png")
    plt.show()
    print("2D Analysis saved to images/motion_2d_analysis.png")
    
    plt.tight_layout()
    plt.savefig("images/motion_analysis.png")
    plt.show()

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    analyze_motion(PATH, num_samples=500)
