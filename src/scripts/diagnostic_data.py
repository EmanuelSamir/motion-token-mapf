import pickle
import os
import numpy as np

data_dir = 'data/trajectories/hdv_1000_ep_abs_coords'
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} not found")
    exit(1)

files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
if not files:
    print("No files found")
    exit(1)

with open(os.path.join(data_dir, files[0]), 'rb') as f:
    data = pickle.load(f)

# Metadata check
print("Metadata:", data.get('metadata', {}))
env_config = data.get('metadata', {}).get('env_config', {})
print(f"Policy Freq: {env_config.get('policy_frequency')}")
print(f"Sim Freq: {env_config.get('simulation_frequency')}")

# Trajectory check
agents = data.get('agents', {})
if not agents:
    print("No agents in data")
    exit(1)

aid = list(agents.keys())[0]
traj = agents[aid]
print(f"Agent {aid} trajectory length: {len(traj)}")
if len(traj) > 1:
    dt_effective = traj[1]['step'] - traj[0]['step']
    print(f"Effective Step Stride in PKL: {dt_effective}")
    p0 = np.array(traj[0]['position'])
    p1 = np.array(traj[1]['position'])
    v0 = np.array(traj[0]['velocity'])
    print(f"P0: {p0}, P1: {p1}, Delta: {p1-p0}")
    print(f"V0: {v0}, Integrated Displacement: {v0 * 0.2}") # Assuming 5Hz policy
