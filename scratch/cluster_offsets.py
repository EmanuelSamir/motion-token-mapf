import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from src.data.hf_dataset import HFTrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

def cluster_offsets(path, num_samples=2000, n_clusters=32):
    dataset = HFTrajectoryDataset(path)
    # Temporary tokenizer to get grid params
    temp_tokenizer = MotionTokenizer()
    
    dt = 0.2
    all_offsets = []
    
    print(f"Sampling {num_samples} items for clustering...")
    for i in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[i]
        gt_futures = sample['gt_future'].numpy()
        masks = sample['m_mask'].numpy()
        init_deltas = sample['initial_deltas'].numpy()
        
        for n in range(gt_futures.shape[0]):
            if masks[n].sum() < 3: continue
            
            traj_gt = gt_futures[n][masks[n] > 0]
            v = np.diff(traj_gt, axis=0) # [T-1, 2]
            v_seq = np.vstack([init_deltas[n], v]) # [T, 2]
            
            # Change in velocity per step (displacements)
            dv = np.diff(v_seq, axis=0) # [T-1, 2]
            
            # Map to index offsets
            off_x = dv[:, 0] / temp_tokenizer.v_step_x
            off_y = dv[:, 1] / temp_tokenizer.v_step_y
            
            all_offsets.append(np.stack([off_x, off_y], axis=1))

    all_offsets = np.concatenate(all_offsets, axis=0)
    print(f"Total acceleration points sampled: {len(all_offsets)}")

    # Remove duplicates to avoid K-Means collapsing into the (0,0) peak if it's too dominant
    # or use a smaller weights for the peak.
    # Actually, let's just use unique integer-rounded offsets to see the 'shape' of possible actions
    unique_offsets, counts = np.unique(np.round(all_offsets).astype(int), axis=0, return_counts=True)
    print(f"Unique integer offsets found: {len(unique_offsets)}")

    # We want K-Means to follow the density but also reach the tails.
    # We can use the unique offsets weighted by their counts, or just cluster the unique ones to ensure coverage.
    # Clustering the unique ones ensures we have tokens for rare events.
    print(f"Clustering {len(unique_offsets)} unique offset patterns into {n_clusters} tokens...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(unique_offsets)
    
    codebook = np.round(kmeans.cluster_centers_).astype(int)
    # Ensure (0,0) is always in the codebook
    if not any((codebook == [0, 0]).all(axis=1)):
        # Replace the closest cluster to (0,0) with (0,0)
        dists = np.sum(codebook**2, axis=1)
        codebook[np.argmin(dists)] = [0, 0]
        
    codebook = np.unique(codebook, axis=0) # Remove any potential  duplicates after rounding
    
    print("\n--- Optimized Codebook (X offset, Y offset) ---")
    print(codebook.tolist())
    print(f"Final Vocab Size: {len(codebook)}")

    # Plot original points and clusters
    plt.figure(figsize=(10, 10))
    plt.scatter(unique_offsets[:, 0], unique_offsets[:, 1], c='gray', alpha=0.3, s=10, label='Data pts')
    plt.scatter(codebook[:, 0], codebook[:, 1], c='red', marker='x', s=50, label='Tokens')
    plt.title(f"Optimized 2D Token Codebook (K={len(codebook)})")
    plt.xlabel("Index Offset X")
    plt.ylabel("Index Offset Y")
    plt.axvline(0, color='black', alpha=0.3)
    plt.axhline(0, color='black', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig("images/optimized_codebook.png")
    plt.show()

    return codebook

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    codebook = cluster_offsets(PATH)
