import numpy as np
import torch
from src.data.components.vocabulary import MotionLMVocabulary

def test_axis_mapping():
    print("🚀 Testing MotionLM Axis Mapping (Swapped Center to Token 0)...")
    vocab = MotionLMVocabulary(bins_per_axis=13)
    
    # Bin index 6 (zero offset) should map to Token 0
    assert vocab._map_axis_to_vocab(6) == 0
    assert vocab._map_vocab_to_axis(0) == 6
    
    # Bin 0 should map to Token 6 (our swap partner)
    assert vocab._map_axis_to_vocab(0) == 6
    assert vocab._map_vocab_to_axis(6) == 0
    
    # Other bins should map to themselves
    assert vocab._map_axis_to_vocab(1) == 1
    assert vocab._map_vocab_to_axis(1) == 1
    
    print("✅ Axis mapping identities passed.")

def test_nonzero_constant_velocity_to_zero():
    print("\n🚀 Testing Non-Zero Constant Velocity -> Separate Axis Token 0...")
    vocab = MotionLMVocabulary(max_delta=18.0, num_delta_bins=128, bins_per_axis=13)
    
    # Initial velocity is (5.0, -2.0)
    # We will pick a velocity that is exactly a bin center to avoid drift
    # Bin center for idx 82 is -18 + 82 * 0.28125 = 5.0625
    # Bin center for idx 57 is -18 + 57 * 0.28125 = -1.96875
    vx, vy = 5.0625, -1.96875
    
    h_minus_1 = np.array([[[0.0, 0.0]]])
    h_0 = np.array([[[vx, vy]]])
    # Constant velocity: each step adds (vx, vy)
    waypoints = np.zeros((1, 10, 2))
    for t in range(10):
        waypoints[0, t, 0] = vx * (t + 2)
        waypoints[0, t, 1] = vy * (t + 2)
    
    tokens = vocab.quantize_deltas_verlet(waypoints, h_0, h_minus_1)
    
    print(f"Tokens for constant {vx, vy}m/s: {tokens[0, 0, :]}")
    
    # Should all be zero because velocity offset is zero
    assert np.all(tokens == 0)
    print("✅ Non-zero constant velocity correctly maps to Token 0.")

def test_constant_velocity_to_zero():
    print("\n🚀 Testing Constant (Zero) Velocity -> Separate Axis Token 0...")
    vocab = MotionLMVocabulary(max_delta=18.0, num_delta_bins=128, bins_per_axis=13)
    
    # Static agent: all positions are [0,0]
    h_minus_1 = np.array([[[0.0, 0.0]]])
    h_0 = np.array([[[0.0, 0.0]]])
    waypoints = np.zeros((1, 10, 2))
    
    tokens = vocab.quantize_deltas_verlet(waypoints, h_0, h_minus_1)
    # tokens shape: [1, 10, 2]
    
    print(f"First step tokens (X, Y): {tokens[0, 0, :]}")
    
    # Both axes for all steps should be 0
    assert np.all(tokens == 0)
    print("✅ Zero velocity correctly maps to Token 0 for both axes.")

def test_reconstruction_accuracy():
    print("\n🚀 Testing Reconstruction Accuracy (Separated Axes)...")
    vocab = MotionLMVocabulary(max_delta=18.0, num_delta_bins=128, bins_per_axis=13)
    
    # Create an arbitrary smooth trajectory
    t = np.linspace(0, 1, 10)
    waypoints = np.stack([5 * t, -2 * t**2], axis=-1) # [10, 2]
    waypoints = waypoints[np.newaxis, ...] # [1, 10, 2]
    
    # Estimate h1, h2 from start for Verlet logic
    h_0 = waypoints[:, 0, :]
    h_minus_1 = h_0 - (waypoints[:, 1, :] - waypoints[:, 0, :])
    
    tokens = vocab.quantize_deltas_verlet(waypoints, h_0, h_minus_1)
    assert tokens.shape == (1, 10, 2)
    
    reconstructed = vocab.reconstruct(tokens, h_0, h_minus_1)
    
    diff = np.abs(reconstructed.numpy() - waypoints)
    max_err = np.max(diff)
    print(f"Max reconstruction error: {max_err:.4f}m")
    assert max_err < 0.5
    print("✅ Reconstruction accuracy verified.")

import matplotlib.pyplot as plt

def test_visualization():
    print("\n🚀 Visualizing Trajectory Reconstruction (Separated Axes)...")
    vocab = MotionLMVocabulary(max_delta=18.0, num_delta_bins=128, bins_per_axis=13)
    
    # Generate a smooth curve trajectory
    t = np.linspace(0, 4, 16) # 16 steps
    x = 10 * np.sin(0.5 * t)
    y = 5 * t
    waypoints = np.stack([x, y], axis=-1)
    
    dt = 0.5
    h_0 = np.array([[x[0], y[0]]])
    v0 = (waypoints[1] - waypoints[0]) / 1.0
    h_minus_1 = h_0 - v0
    
    waypoints_batched = waypoints[np.newaxis, ...] # [1, 16, 2]
    h_0_batched = h_0[np.newaxis, ...]
    h_minus_1_batched = h_minus_1[np.newaxis, ...]
    
    # Quantize and Reconstruct
    tokens = vocab.quantize_deltas_verlet(waypoints_batched, h_0_batched, h_minus_1_batched)[0]
    # tokens is [16, 2]
    
    reconstructed = vocab.reconstruct(tokens[np.newaxis, ...], h_0_batched, h_minus_1_batched)[0].numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'g-', alpha=0.3, label='Ground Truth')
    
    for i, (p, t_pair) in enumerate(zip(reconstructed, tokens)):
        # If both are zero, blue, else red
        color = 'blue' if np.all(t_pair == 0) else 'red'
        plt.plot(p[0], p[1], color=color, marker='o', markersize=4)
        plt.text(p[0]+0.1, p[1]+0.1, f"({t_pair[0]},{t_pair[1]})", fontsize=7, color=color)
        
    plt.legend()
    plt.title("MotionLM: Separated Axis Reconstruction")
    plt.axis("equal")
    
    save_path = "/Users/emanuelsamir/Documents/dev/cmu/courses/4_multi_robot/project/reconstruction_test_separated.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 Visualization saved to: {save_path}")

def test_visualize_velocity_grid():
    print("\n🚀 Visualizing the 169 Velocity Offsets...")
    vocab = MotionLMVocabulary(max_delta=18.0, num_delta_bins=128, bins_per_axis=13)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # --- Panel 1: 1D Token -> Offset Mapping ---
    tokens = np.arange(13)
    bins = [vocab._map_vocab_to_axis(t) for t in tokens]
    offsets = [b - 6 for b in bins]
    
    ax1.step(tokens, offsets, where='mid', marker='o', color='purple')
    ax1.set_xticks(tokens)
    ax1.set_yticks(np.arange(-6, 7))
    ax1.grid(True, alpha=0.3)
    ax1.set_title("1D Mapping: Token -> Index Offset")
    ax1.set_xlabel("Vocabulary Token (0-12)")
    ax1.set_ylabel("Index Offset (±6)")
    for i, txt in enumerate(offsets):
        ax1.annotate(f"{txt}", (tokens[i], offsets[i]+0.2), ha='center', fontsize=8)

    # --- Panel 2: 2D Velocity Neighborhood (13x13) ---
    curr_idx_x, curr_idx_y = 64, 64 # Center (0.0 m/s)
    base_vx = vocab._delta_idx_to_val(curr_idx_x)
    base_vy = vocab._delta_idx_to_val(curr_idx_y)
    
    # Background discretization (subset)
    bg_range = np.arange(55, 75)
    bg_vals = [vocab._delta_idx_to_val(i) for i in bg_range]
    for bv in bg_vals:
        ax2.axvline(bv, color='gray', alpha=0.1, linewidth=0.5)
        ax2.axhline(bv, color='gray', alpha=0.1, linewidth=0.5)
        
    for tx in range(13):
        for ty in range(13):
            # Calculate physical velocity for this token pair
            bx = vocab._map_vocab_to_axis(tx)
            by = vocab._map_vocab_to_axis(ty)
            
            vx = vocab._delta_idx_to_val(curr_idx_x + (bx - 6))
            vy = vocab._delta_idx_to_val(curr_idx_y + (by - 6))
            
            is_center = (tx == 0 and ty == 0)
            color = 'blue' if is_center else 'red'
            size = 40 if is_center else 15
            
            ax2.scatter(vx, vy, c=color, s=size, alpha=0.6)
            # Annotate a few points for clarity
            if tx % 4 == 0 and ty % 4 == 0:
                ax2.annotate(f"({tx},{ty})", (vx, vy), fontsize=7, alpha=0.8)
    
    ax2.set_title(f"Reachable Velocities from V=({base_vx:.1f}, {base_vy:.1f})")
    ax2.set_xlabel("Next Velocity X (m/s)")
    ax2.set_ylabel("Next Velocity Y (m/s)")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.1)
    
    plt.tight_layout()
    save_path = "/Users/emanuelsamir/Documents/dev/cmu/courses/4_multi_robot/project/velocity_grid_visualization.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 Enhanced Velocity grid visualization saved to: {save_path}")

def test_visualize_full_delta_space():
    print("\n🚀 Visualizing the Full 128-Bin Delta Space vs 13 Verlet Offsets...")
    vocab = MotionLMVocabulary(max_delta=18.0, num_delta_bins=128, bins_per_axis=13)
    
    # Generate all 128 bin centers
    all_bins = np.arange(128)
    all_vals = [vocab._delta_idx_to_val(i) for i in all_bins]
    
    plt.figure(figsize=(15, 4))
    
    # 1. Plot all 128 bins as small background dots
    plt.scatter(all_vals, np.zeros_like(all_vals), c='gray', s=5, alpha=0.3, label='Full 128-bin Space (-18m to 18m)')
    
    # 2. Pick a "Current Velocity" bin (e.g. index 40, which is around -6.7 m/s)
    curr_idx = 40
    curr_val = vocab._delta_idx_to_val(curr_idx)
    plt.scatter([curr_val], [0], c='green', s=100, marker='|', label=f'Current Velocity (Idx {curr_idx})', zorder=5)
    
    # 3. Identify the 13 reachable bins via Verlet offsets [-6, +6]
    reachable_indices = np.clip(np.arange(curr_idx - 6, curr_idx + 7), 0, 127)
    reachable_vals = [vocab._delta_idx_to_val(i) for i in reachable_indices]
    
    # Highlight reachable bins
    plt.scatter(reachable_vals, np.zeros_like(reachable_vals), c='red', s=30, edgecolors='black', label='Reachable via 13 Tokens (±6 bins)')
    
    # Add annotations
    plt.annotate('Constant Velocity (Token 0)', (curr_val, 0.02), ha='center', color='green', fontsize=9, fontweight='bold')
    plt.annotate('Min Delta (-18m)', (-18, -0.02), ha='center', fontsize=8)
    plt.annotate('Max Delta (+18m)', (18, -0.02), ha='center', fontsize=8)
    
    plt.title("MotionLM Vocabulary: 128 Total Bins vs 13 Searchable Tokens per Step")
    plt.yticks([]) # Hide Y axis
    plt.xlabel("Velocity Delta (m/s)")
    plt.ylim(-0.1, 0.1)
    plt.legend(loc='upper right')
    plt.grid(True, axis='x', alpha=0.1)
    
    save_path = "/Users/emanuelsamir/Documents/dev/cmu/courses/4_multi_robot/project/delta_space_128_bins.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 128-bin delta space visualization saved to: {save_path}")

if __name__ == "__main__":
    test_axis_mapping()
    test_constant_velocity_to_zero()
    test_nonzero_constant_velocity_to_zero()
    test_reconstruction_accuracy()
    test_visualization()
    test_visualize_velocity_grid()
    test_visualize_full_delta_space()
