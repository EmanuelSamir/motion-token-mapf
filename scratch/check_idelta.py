from src.data.hf_dataset import HFTrajectoryDataset
import numpy as np

def inspect_sample_ids(path, sample_idx=265, agent_idx=3):
    dataset = HFTrajectoryDataset(path)
    item = dataset[sample_idx]
    
    history = item['history'][agent_idx].numpy()
    init_delta = item['initial_deltas'][agent_idx].numpy()
    gt_future = item['gt_future'][agent_idx].numpy()
    mask = item['m_mask'][agent_idx].numpy()
    
    print(f"Sample {sample_idx}, Agent {agent_idx}")
    print(f"History (last 3 steps):\n{history[-3:]}")
    print(f"Initial Delta: {init_delta}")
    print(f"GT Future (first 3 steps):\n{gt_future[:3]}")
    
    # Calculate what it should be
    p_now = history[-1, :2]
    p_prev = history[-2, :2]
    calc_delta = p_now - p_prev
    print(f"Calculated Delta (from history): {calc_delta}")
    
    if np.any(np.abs(calc_delta - init_delta) > 1e-5):
        print("Mismatch between stored and calculated delta!")

if __name__ == "__main__":
    PATH = 'data/trajectories/run_20260423_015507'
    inspect_sample_ids(PATH)
