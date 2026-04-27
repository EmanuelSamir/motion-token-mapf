import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from src.utils.tokenizer import MotionTokenizer

def visualize_token_grid():
    tokenizer = MotionTokenizer()
    
    os.makedirs("images", exist_ok=True)
    
    # Create a grid of all possible tokens
    # Token T = off_idx_x * num_bins_y + off_idx_y
    
    grid_offsets_x = []
    grid_offsets_y = []
    
    for t in range(tokenizer.vocab_size):
        off_idx_x = t // tokenizer.num_bins_y
        off_idx_y = t % tokenizer.num_bins_y
        
        grid_offsets_x.append(tokenizer.offsets_x[off_idx_x])
        grid_offsets_y.append(tokenizer.offsets_y[off_idx_y])
        
    grid_offsets_x = np.array(grid_offsets_x)
    grid_offsets_y = np.array(grid_offsets_y)
    
    # Convert index offsets to physical accelerations [m/s^2]
    # accel = offset * v_step_m_per_step / (dt^2)
    accel_x = grid_offsets_x * tokenizer.v_step_x / (tokenizer.dt**2)
    accel_y = grid_offsets_y * tokenizer.v_step_y / (tokenizer.dt**2)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(accel_x, accel_y, c='blue', marker='s', s=100, alpha=0.6)
    
    for i, txt in enumerate(range(tokenizer.vocab_size)):
        plt.annotate(txt, (accel_x[i], accel_y[i]), fontsize=8, ha='center', va='center', color='white')
        
    plt.title(f"Tokenizer Acceleration Grid (Total Tokens: {tokenizer.vocab_size})\nAsymmetric & Non-Homogeneous")
    plt.xlabel("Acceleration X [m/s^2]")
    plt.ylabel("Acceleration Y [m/s^2]")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', lw=1)
    plt.axhline(0, color='black', lw=1)
    
    plt.savefig("images/token_grid.png")
    plt.show()
    
    print(f"Token grid saved to images/token_grid.png")

if __name__ == "__main__":
    visualize_token_grid()
