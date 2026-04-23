import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence, Array2D, Array3D
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.tokenizer import MotionTokenizer
from src.data.dataset import TrajectoryDataset

def convert_dataset_to_hf(data_dir, output_path, config=None):
    if config is None:
        config = {
            'history_len': 10,
            'prediction_len': 25,
            'neighbor_radius': 50.0,
            'stride': 5,
            'stationary_keep_ratio': 0.1,
            'max_agents_per_scene': 20,
            'sample_ratio': 1.0
        }
    
    # Initialize basic dataset to leverage its sample generation logic
    # but we will manually process to avoid RAM issues if needed
    # Actually, we can just use TrajectoryDataset's _load_samples if we modify it slightly 
    # to not include agents_map in the returned samples, or just use it as is for now 
    # since we are doing a one-time conversion.
    
    tokenizer = MotionTokenizer()
    
    print(f"Initializing temporary dataset for sample discovery...")
    temp_ds = TrajectoryDataset(
        data_dir=data_dir,
        history_len=config['history_len'],
        prediction_len=config['prediction_len'],
        neighbor_radius=config['neighbor_radius'],
        tokenizer=tokenizer,
        stride=config['stride'],
        sample_ratio=config['sample_ratio'],
        stationary_keep_ratio=config['stationary_keep_ratio'],
        max_agents_per_scene=config['max_agents_per_scene'],
        use_cache=True # Use existing cache if available to speed up discovery
    )
    
    total_samples = len(temp_ds)
    print(f"Processing {total_samples} samples into HF format...")
    
    def gen():
        for i in tqdm(range(total_samples)):
            # Use the existing __getitem__ logic to get processed tensors
            item = temp_ds[i]
            
            # Convert tensors to numpy for HF compatibility
            yield {
                'history': item['history'].numpy(),
                'tokens': item['tokens'].numpy(),
                'agent_ids': item['agent_ids'].numpy(),
                'time_ids': item['time_ids'].numpy(),
                'gt_future': item['gt_future'].numpy(),
                'm_mask': item['m_mask'].numpy(),
                'initial_deltas': item['initial_deltas'].numpy(),
                'num_agents': int(item['num_agents'])
            }

    # Define features for structured storage
    h = config['history_len']
    p = config['prediction_len']
    m = config['max_agents_per_scene']
    
    features = Features({
        'history': Array3D(shape=(m, h, 5), dtype='float32'),
        'tokens': Sequence(Value('int64')),
        'agent_ids': Sequence(Value('int64')),
        'time_ids': Sequence(Value('int64')),
        'gt_future': Array3D(shape=(m, p + 1, 2), dtype='float32'),
        'm_mask': Array2D(shape=(m, p + 1), dtype='float32'),
        'initial_deltas': Array2D(shape=(m, 2), dtype='float32'),
        'num_agents': Value('int64')
    })
    
    dataset = Dataset.from_generator(gen, features=features)
    
    print(f"Saving dataset to {output_path}...")
    dataset.save_to_disk(output_path)
    print("Conversion complete!")

if __name__ == "__main__":
    # You can customize these paths or read from hydra config
    DATA_DIR = "data/trajectories/hdv_1000_ep_abs_coords"
    OUTPUT_PATH = "data/trajectories/hdv_1000_ep_hf_v20"
    
    convert_dataset_to_hf(DATA_DIR, OUTPUT_PATH)
