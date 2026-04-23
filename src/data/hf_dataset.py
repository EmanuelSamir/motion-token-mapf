from datasets import load_from_disk
import torch
from torch.utils.data import Dataset

class HFTrajectoryDataset(Dataset):
    """
    Dataset that loads pre-processed trajectories from a Hugging Face Dataset on disk.
    Uses memory-mapping (Arrow) to keep RAM usage low and access fast.
    """
    def __init__(self, path, sample_ratio=1.0, seed=42):
        self.dataset = load_from_disk(path)
        if sample_ratio < 1.0:
            num_samples = int(len(self.dataset) * sample_ratio)
            # Use deterministic selection for consistency
            import numpy as np
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.dataset), size=num_samples, replace=False)
            # Sort indices to maintain some locality for memory-mapping efficiency
            indices = sorted(indices.tolist())
            self.dataset = self.dataset.select(indices)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Convert the dictionary of lists/arrays to a dictionary of torch tensors
        return {
            'history': torch.tensor(item['history'], dtype=torch.float32),
            'tokens': torch.tensor(item['tokens'], dtype=torch.long),
            'agent_ids': torch.tensor(item['agent_ids'], dtype=torch.long),
            'time_ids': torch.tensor(item['time_ids'], dtype=torch.long),
            'gt_future': torch.tensor(item['gt_future'], dtype=torch.float32),
            'm_mask': torch.tensor(item['m_mask'], dtype=torch.float32),
            'initial_deltas': torch.tensor(item['initial_deltas'], dtype=torch.float32),
            'num_agents': torch.tensor(item['num_agents'], dtype=torch.long)
        }
