
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from src.utils.visualizer import plot_comparison

def inspect_sample(file_path):
    print(f"Inspecting {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    roadgraph = data["roadgraph"]
    trajectories = data["trajectories"]
    
    print(f"Roadgraph points: {len(roadgraph)}")
    print(f"Time steps recorded: {len(trajectories)}")
    
    # Prepare data for visualizer
    # visualizer expects: history, gt_future, pred_future, roadgraph, interactions
    # Let's just plot the whole trajectory of first 2 agents as "GT"
    
    all_agents = {}
    for t, agents in enumerate(trajectories):
        for agent in agents:
            aid = agent["id"]
            if aid not in all_agents:
                all_agents[aid] = []
            all_agents[aid].append(agent["pos"])
            
    # Convert to numpy [num_agents, T, 2]
    num_steps = len(trajectories)
    agent_ids = list(all_agents.keys())[:2] # Just show first 2 for now
    
    gt_future = []
    for aid in agent_ids:
        traj = np.array(all_agents[aid])
        if len(traj) < num_steps:
            # Pad
            traj = np.pad(traj, ((0, num_steps - len(traj)), (0, 0)), mode='edge')
        gt_future.append(traj)
    
    gt_future = np.stack(gt_future)
    history = gt_future[:, :2, :] # Dummy history
    pred_future = gt_future
    
    # Extract control status per agent
    is_controlled = []
    for i in range(num_agents):
        # Check first step for control status
        is_controlled.append(data["trajectories"][0][i].get("is_controlled", False))
    
    # 4. Visualize first few agents
    fig = plot_comparison(
        history, gt_future, pred_future, roadgraph, 
        is_controlled=is_controlled,
        title=f"Scenario: {os.path.basename(file_path)}"
    )
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_sample(sys.argv[1])
    else:
        # Try a default file
        default_file = "data/highway_raw/highway-v0_00000.pkl"
        if os.path.exists(default_file):
            inspect_sample(default_file)
        else:
            print("Please provide a path to a .pkl file.")
