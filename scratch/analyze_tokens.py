import os
import sys

# Ensure src is in path
BASE_DIR = os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import torch
from src.data.hf_dataset import HFTrajectoryDataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(paths, num_samples=5000):
    if isinstance(paths, str):
        paths = [paths]
        
    print(f"--- Analyzing tokens across {len(paths)} runs (target: {num_samples} samples) ---")
    
    token_counts = Counter()
    total_valid_tokens = 0
    samples_per_path = num_samples // len(paths)
    
    for path in paths:
        print(f"  Processing: {path}...")
        try:
            # We use a higher sample ratio to get enough samples from each run
            dataset = HFTrajectoryDataset(path, sample_ratio=1.0)
            if len(dataset) == 0: continue
            
            n = min(len(dataset), samples_per_path)
            for i in range(n):
                batch = dataset[i]
                tokens = batch['tokens'].numpy()
                valid_tokens = tokens[tokens != 169] # 169 is INVALID_TOKEN
                token_counts.update(valid_tokens)
                total_valid_tokens += len(valid_tokens)
        except Exception as e:
            print(f"    Error: {e}")
            
    if total_valid_tokens == 0:
        print("No valid tokens found!")
        return

    # 1. interpretation
    count_84 = token_counts[84]
    pct_84 = (count_84 / total_valid_tokens) * 100
    
    print("\n" + "="*50)
    print(f"STATISTICS FOR {total_valid_tokens} TOKENS")
    print(f"Token 84 (Center/Idle/Const Vel): {count_84} ({pct_84:.2f}%)")
    print("="*50)
    
    # 2. Grid data preparation (13x13)
    grid = np.zeros((13, 13))
    for token, count in token_counts.items():
        if token < 169:
            grid[token // 13, token % 13] = count
    
    grid_pct = (grid / total_valid_tokens) * 100

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # A. Linear scale heatmap
    sns.heatmap(grid_pct, annot=False, cmap="YlGnBu", ax=axes[0])
    axes[0].set_title(f"Linear Density (%)\nDominance of Token 84 (Center)")
    axes[0].set_xlabel("Relative ΔY index (Lane change / Turning)")
    axes[0].set_ylabel("Relative ΔX index (Acceleration / Braking)")
    axes[0].add_patch(plt.Rectangle((6, 6), 1, 1, fill=False, edgecolor='red', lw=4, label="Token 84 (Zero)"))

    # B. Log scale heatmap
    log_grid = np.log10(grid + 1)
    sns.heatmap(log_grid, annot=False, cmap="viridis", ax=axes[1])
    axes[1].set_title("Log10 Density (Counts)\nReveals rare maneuvers (the 'Tail' of distribution)")
    axes[1].set_xlabel("Relative ΔY index")
    axes[1].set_ylabel("Relative ΔX index")
    axes[1].add_patch(plt.Rectangle((6, 6), 1, 1, fill=False, edgecolor='white', lw=2))

    plt.suptitle(f"Token Distribution Analysis\nAnalyzed {total_valid_tokens} valid tokens from {len(paths)} runs", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    viz_path = "token_distribution_full.png"
    plt.savefig(viz_path)
    print(f"\nHeatmap saved to: {viz_path}")
    
    # 4. Interpret specific regions
    print("\nINTERPRETATION OF THE GRID:")
    print("- Center [6, 6]: Idle or Constant Velocity (Token 84)")
    print("- Top [0-5, 6]: Accelerations (Higher ΔX)")
    print("- Bottom [7-12, 6]: Braking/Deceleration (Lower ΔX)")
    print("- Left [6, 0-5]: Left maneuvers (Lane changes)")
    print("- Right [6, 7-12]: Right maneuvers (Lane changes)")

    # 5. Summary
    print("\nSummary:")
    non_84_pct = 100 - pct_84
    print(f"Maneuver Variety (Non-84 tokens): {non_84_pct:.2f}%")
    
    if pct_84 > 90:
        print("🚩 CRITICAL: High imbalance. Use Gamma=5.0 and Alpha ~ 0.5.")
    elif pct_84 > 70:
        print("⚠️ WARNING: Moderate imbalance. Use Gamma=3.0.")
    else:
        print("✅ Good distribution! Gamma=2.0 is likely enough.")

if __name__ == "__main__":
    base_data_path = "data/trajectories"
    if os.path.exists(base_data_path):
        runs = [os.path.join(base_data_path, d) for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))]
        if runs:
            # Analyze ALL available data
            analyze_dataset(runs, num_samples=10000)
        else:
            print(f"No runs found in {base_data_path}")
    else:
        print(f"Path {base_data_path} does not exist.")
