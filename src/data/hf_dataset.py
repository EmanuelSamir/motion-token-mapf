import os
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset
from src.utils.tokenizer import MotionTokenizer
import numpy as np

class HFTrajectoryDataset(Dataset):
    """
    Dataset that loads pre-processed trajectories from a Hugging Face Dataset on disk.
    Uses memory-mapping (Arrow) to keep RAM usage low and access fast.
    """
    def __init__(self, path, sample_ratio=1.0, seed=42, max_agents=3):
        self.dataset = load_from_disk(path)
        self.tokenizer = MotionTokenizer()
        self.max_agents = max_agents
        
        if sample_ratio < 1.0:
            num_samples = int(len(self.dataset) * sample_ratio)
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.dataset), size=num_samples, replace=False)
            indices = sorted(indices.tolist())
            self.dataset = self.dataset.select(indices)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        history = torch.tensor(item['history'], dtype=torch.float32)
        gt_future = torch.tensor(item['gt_future'], dtype=torch.float32)
        masks = torch.tensor(item['m_mask'], dtype=torch.float32)
        
        # We always process exactly self.max_agents
        N = self.max_agents
        T_pred = gt_future.shape[1] - 1
        
        # 1. Derive accurate initial_deltas from recorded velocity
        dt = 0.2
        init_deltas = history[:, -1, 2:4] * dt
        
        # 2. Re-tokenize trajectories using the corrected init_deltas
        all_tokens = []
        for n in range(N):
            if n >= history.shape[0] or masks[n].sum() < 2:
                all_tokens.append([self.tokenizer.INVALID_TOKEN] * T_pred)
                continue
            
            valid_idx = masks[n] > 0
            traj_gt = gt_future[n][valid_idx].numpy()
            idelta_np = init_deltas[n].numpy()
            
            agent_tokens = self.tokenizer.tokenize_trajectory(traj_gt, initial_delta=idelta_np)
            
            padded_tokens = [self.tokenizer.INVALID_TOKEN] * T_pred
            for i, t in enumerate(agent_tokens):
                if i < T_pred:
                    padded_tokens[i] = t
            all_tokens.append(padded_tokens)

        # 3. Interleave tokens (Time-first, Agent-second)
        interleaved_tokens = []
        agent_ids = []
        time_ids = []
        
        for t in range(T_pred):
            for n in range(N):
                interleaved_tokens.append(all_tokens[n][t])
                agent_ids.append(n)
                time_ids.append(t)

        return {
            'history': history,
            'tokens': torch.tensor(interleaved_tokens, dtype=torch.long),
            'agent_ids': torch.tensor(agent_ids, dtype=torch.long),
            'time_ids': torch.tensor(time_ids, dtype=torch.long),
            'gt_future': gt_future,
            'm_mask': masks,
            'initial_deltas': init_deltas,
            'num_agents': torch.tensor(N, dtype=torch.long)
        }
