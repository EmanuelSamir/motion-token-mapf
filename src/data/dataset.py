import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.tokenizer import MotionTokenizer

class TrajectoryDataset(Dataset):
    """
    Dataset for Loading Multi-Agent Trajectories.
    Supports Agent-Centric storage, Ego-centric normalization, and 
    VARIABLE number of agents per scene.
    """
    def __init__(
        self,
        data_dir,
        history_len=10,
        prediction_len=25,
        neighbor_radius=50.0,
        tokenizer=None,
        stride=5,
        sample_ratio=1.0,
        stationary_keep_ratio=0.1,
        max_agents_per_scene=10,
        use_cache=True
    ):
        self.data_dir = data_dir
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.neighbor_radius = neighbor_radius
        self.tokenizer = tokenizer or MotionTokenizer()
        self.stride = stride
        self.sample_ratio = sample_ratio
        self.stationary_keep_ratio = stationary_keep_ratio
        self.INVALID_TOKEN = 170
        self.max_agents_per_scene = max_agents_per_scene
        
        # Unique cache name based on parameters
        param_hash = f"r{neighbor_radius}_s{stride}_h{history_len}_p{prediction_len}_sk{stationary_keep_ratio}_m{max_agents_per_scene}"
        self.cache_path = os.path.join(data_dir, f"samples_cache_{param_hash}.pkl")
        
        self.samples = self._load_samples(use_cache)

    def _load_samples(self, use_cache):
        if use_cache and os.path.exists(self.cache_path):
            print(f"Loading samples from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"Generating samples from {self.data_dir} (Cold start, this may take a few minutes)...")
        samples = []
        if not os.path.exists(self.data_dir):
            print(f"WARNING: Data directory {self.data_dir} not found.")
            return []
            
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.pkl') and f.startswith('hdv_data')])
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            agents_map = data.get('agents', {})
            if not agents_map: continue

            # Pre-index agents for faster lookup
            agents_by_step = {} # {aid: {step: state}}
            max_t = 0
            for aid, states in agents_map.items():
                if not states: continue
                agents_by_step[aid] = {s['step']: s for s in states}
                max_t = max(max_t, states[-1]['step'])

            total_len = self.history_len + self.prediction_len
            
            for start_t in range(0, max_t - total_len, self.stride):
                for ego_id, states in agents_map.items():
                    # Check if Ego has complete history
                    steps_available = agents_by_step[ego_id]
                    if not all(start_t + t in steps_available for t in range(self.history_len)):
                        continue
                    
                    # Stationary oversampling check
                    pos_start = steps_available[start_t]['position']
                    pos_end = steps_available[start_t + self.history_len - 1]['position']
                    dist = np.linalg.norm(np.array(pos_end) - np.array(pos_start))
                    
                    if dist < 1.0 and np.random.random() > self.stationary_keep_ratio:
                        continue
                        
                    t_ref = start_t + self.history_len - 1
                    ego_pos = np.array(steps_available[t_ref]['position'])
                    
                    # Collect neighbors
                    nearby_neighbors = []
                    for aid, step_map in agents_by_step.items():
                        if aid == ego_id: continue
                        if t_ref in step_map:
                            ref_pos = np.array(step_map[t_ref]['position'])
                            d = np.linalg.norm(ref_pos - ego_pos)
                            if d < self.neighbor_radius:
                                nearby_neighbors.append((aid, d))
                    
                    nearby_neighbors.sort(key=lambda x: x[1])
                    selected_neighbors = [n[0] for n in nearby_neighbors[:self.max_agents_per_scene - 1]]
                    
                    group = [ego_id] + selected_neighbors
                    
                    samples.append({
                        'file': file,
                        'start_t': start_t,
                        'agents': group,
                        'agents_map': agents_map # This keeps the full dict, which is fine for pickle
                    })
        
        if self.sample_ratio < 1.0:
            import random
            random.shuffle(samples)
            samples = samples[:int(len(samples)*self.sample_ratio)]
            
        print(f"Dataset Loaded: {len(samples)} valid scenes. Saving cache...")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(samples, f)
            
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        group, agents_map, start_t = s['agents'], s['agents_map'], s['start_t']
        num_scene_agents = len(group)
        
        # Reference time for history boundary
        t_ref = start_t + self.history_len - 1

        histories, futures, init_deltas, masks = [], [], [], []
        
        for aid in group:
            aid_states = agents_map.get(aid, [])
            states_dict = {st['step']: st for st in aid_states}
            
            h_feats = []
            for t in range(start_t, start_t + self.history_len):
                if t in states_dict:
                    v = states_dict[t]
                    # ABSOLUTE coordinates used for map-awareness (e.g., Y-pos for Ramp)
                    pos = np.array(v['position'])
                    vel = np.array(v['velocity'])
                    yaw = v['heading']
                    h_feats.append([pos[0], pos[1], vel[0], vel[1], yaw])
                else:
                    h_feats.append([0.0, 0.0, 0.0, 0.0, 0.0])
            
            h_feats = np.array(h_feats)
            histories.append(h_feats)
            
            f_coords, f_mask = [], []
            # Reference for tokenizer (Verlet initialization)
            p_now, p_prev = h_feats[-1, :2], h_feats[-2, :2]
            idelta = p_now - p_prev
            init_deltas.append(idelta)
            
            for t in range(t_ref, t_ref + self.prediction_len + 1):
                if t in states_dict:
                    v = states_dict[t]
                    pos = np.array(v['position'])
                    f_coords.append(pos)
                    f_mask.append(1)
                else:
                    f_coords.append(f_coords[-1] if f_coords else [0.0, 0.0])
                    f_mask.append(0)
            
            futures.append(np.array(f_coords))
            masks.append(np.array(f_mask))

        all_tokens = []
        for f, idelta, m in zip(futures, init_deltas, masks):
            # Tokenizer handles the relative delta-deltas, 
            # so absolute coordinate input is fine.
            tokens = self.tokenizer.tokenize_trajectory(f, initial_delta=idelta)
            for i in range(len(tokens)):
                if m[i+1] == 0:
                    tokens[i] = self.INVALID_TOKEN
            all_tokens.append(tokens)

        # Interleave tokens
        interleaved = []
        for t in range(self.prediction_len):
            for n in range(num_scene_agents):
                interleaved.append(all_tokens[n][t])
            for n in range(num_scene_agents, self.max_agents_per_scene):
                interleaved.append(self.INVALID_TOKEN)

        # Pad Tensors to max_agents_per_scene
        history_tensor = torch.zeros((self.max_agents_per_scene, self.history_len, 5), dtype=torch.float32)
        history_tensor[:num_scene_agents] = torch.tensor(np.stack(histories), dtype=torch.float32)

        gt_future_tensor = torch.zeros((self.max_agents_per_scene, self.prediction_len + 1, 2), dtype=torch.float32)
        gt_future_tensor[:num_scene_agents] = torch.tensor(np.stack(futures), dtype=torch.float32)

        mask_tensor = torch.zeros((self.max_agents_per_scene, self.prediction_len + 1), dtype=torch.float32)
        mask_tensor[:num_scene_agents] = torch.tensor(np.stack(masks), dtype=torch.float32)

        init_deltas_tensor = torch.zeros((self.max_agents_per_scene, 2), dtype=torch.float32)
        init_deltas_tensor[:num_scene_agents] = torch.tensor(np.stack(init_deltas), dtype=torch.float32)
        
        return {
            'history': history_tensor,
            'tokens': torch.tensor(interleaved, dtype=torch.long),
            'agent_ids': torch.tensor(np.tile(np.arange(self.max_agents_per_scene), self.prediction_len), dtype=torch.long),
            'time_ids': torch.tensor(np.repeat(np.arange(self.prediction_len), self.max_agents_per_scene), dtype=torch.long),
            'gt_future': gt_future_tensor,
            'm_mask': mask_tensor, 
            'initial_deltas': init_deltas_tensor,
            'num_agents': torch.tensor(num_scene_agents, dtype=torch.long)
        }
