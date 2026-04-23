import os
import numpy as np
import torch
from datasets import Dataset, Features, Value, Sequence, Array2D, Array3D
from src.utils.tokenizer import MotionTokenizer
from typing import Dict, List, Any

class HFDatasetWriter:
    """
    Handles on-the-fly conversion of raw simulation trajectories 
    into tokenized datasets for Hugging Face compatibility.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'history_len': config.get('history_len', 10),
            'prediction_len': config.get('prediction_len', 25),
            'neighbor_radius': config.get('neighbor_radius', 50.0),
            'stride': config.get('stride', 5),
            'stationary_keep_ratio': config.get('stationary_keep_ratio', 0.1),
            'max_agents_per_scene': config.get('max_agents', 20),
        }
        self.tokenizer = MotionTokenizer()
        self.samples = []
        self.INVALID_TOKEN = 170

    def process_episode(self, agents_map: Dict[int, List[Dict]], roadgraph: List[Dict]):
        """
        Converts a raw episode (after simulation graduation) into multiple 
        sliding-window samples for the Dataset.
        """
        if not agents_map:
            return 0

        # 1. Pre-index agents for faster lookup
        agents_by_step = {}
        max_t = 0
        for aid, states in agents_map.items():
            if not states: continue
            agents_by_step[aid] = {s['step']: s for s in states}
            max_t = max(max_t, states[-1]['step'])

        total_len = self.config['history_len'] + self.config['prediction_len']
        new_samples_count = 0

        # 2. Extract windows
        for start_t in range(0, max_t - total_len, self.config['stride']):
            for ego_id, states in agents_map.items():
                # Check if Ego has complete history
                steps_available = agents_by_step[ego_id]
                if not all(start_t + t in steps_available for t in range(self.config['history_len'])):
                    continue
                
                # Stationary oversampling check
                pos_start = steps_available[start_t]['position']
                pos_end = steps_available[start_t + self.config['history_len'] - 1]['position']
                dist = np.linalg.norm(np.array(pos_end) - np.array(pos_start))
                if dist < 1.0 and np.random.random() > self.config['stationary_keep_ratio']:
                    continue
                    
                t_ref = start_t + self.config['history_len'] - 1
                ego_pos = np.array(steps_available[t_ref]['position'])
                
                # Collect neighbors
                nearby_neighbors = []
                for aid, step_map in agents_by_step.items():
                    if aid == ego_id: continue
                    if t_ref in step_map:
                        ref_pos = np.array(step_map[t_ref]['position'])
                        d = np.linalg.norm(ref_pos - ego_pos)
                        if d < self.config['neighbor_radius']:
                            nearby_neighbors.append((aid, d))
                
                nearby_neighbors.sort(key=lambda x: x[1])
                selected_neighbors = [n[0] for n in nearby_neighbors[:self.config['max_agents_per_scene'] - 1]]
                group = [ego_id] + selected_neighbors
                
                # 3. Create actual tensorized sample (logic from TrajectoryDataset.__getitem__)
                sample = self._create_sample(group, agents_by_step, start_t)
                self.samples.append(sample)
                new_samples_count += 1
        
        return new_samples_count

    def _create_sample(self, group: List[int], agents_by_step: Dict, start_t: int) -> Dict[str, Any]:
        """Process a specific group of agents into a sample dictionary."""
        num_scene_agents = len(group)
        t_ref = start_t + self.config['history_len'] - 1
        
        histories, futures, init_deltas, masks = [], [], [], []
        
        for aid in group:
            step_map = agents_by_step[aid]
            
            h_feats = []
            for t in range(start_t, start_t + self.config['history_len']):
                if t in step_map:
                    v = step_map[t]
                    pos, vel, yaw = v['position'], v['velocity'], v['heading']
                    h_feats.append([pos[0], pos[1], vel[0], vel[1], yaw])
                else:
                    h_feats.append([0.0, 0.0, 0.0, 0.0, 0.0])
            
            h_feats = np.array(h_feats)
            histories.append(h_feats)
            
            f_coords, f_mask = [], []
            p_now, p_prev = h_feats[-1, :2], h_feats[-2, :2]
            init_deltas.append(p_now - p_prev)
            
            for t in range(t_ref, t_ref + self.config['prediction_len'] + 1):
                if t in step_map:
                    pos = np.array(step_map[t]['position'])
                    f_coords.append(pos)
                    f_mask.append(1)
                else:
                    f_coords.append(f_coords[-1] if f_coords else [0.0, 0.0])
                    f_mask.append(0)
            
            futures.append(np.array(f_coords))
            masks.append(np.array(f_mask))

        # Tokenization
        all_tokens = []
        for f, idelta, m in zip(futures, init_deltas, masks):
            tokens = self.tokenizer.tokenize_trajectory(f, initial_delta=idelta)
            for i in range(len(tokens)):
                if m[i+1] == 0:
                    tokens[i] = self.INVALID_TOKEN
            all_tokens.append(tokens)

        # Interleaving
        interleaved = []
        for t in range(self.config['prediction_len']):
            for n in range(num_scene_agents):
                interleaved.append(int(all_tokens[n][t]))
            for n in range(num_scene_agents, self.config['max_agents_per_scene']):
                interleaved.append(self.INVALID_TOKEN)

        return {
            'history': np.pad(np.stack(histories), ((0, self.config['max_agents_per_scene'] - num_scene_agents), (0,0), (0,0))),
            'tokens': interleaved,
            'agent_ids': np.tile(np.arange(self.config['max_agents_per_scene']), self.config['prediction_len']).tolist(),
            'time_ids': np.repeat(np.arange(self.config['prediction_len']), self.config['max_agents_per_scene']).tolist(),
            'gt_future': np.pad(np.stack(futures), ((0, self.config['max_agents_per_scene'] - num_scene_agents), (0,0), (0,0))),
            'm_mask': np.pad(np.stack(masks), ((0, self.config['max_agents_per_scene'] - num_scene_agents), (0,0))),
            'initial_deltas': np.pad(np.stack(init_deltas), ((0, self.config['max_agents_per_scene'] - num_scene_agents), (0,0))),
            'num_agents': num_scene_agents
        }

    def save(self, path: str):
        """Finalizes the dataset and saves to disk."""
        if not self.samples:
            print("WARNING: No samples collected. Skipping save.")
            return

        m = self.config['max_agents_per_scene']
        h = self.config['history_len']
        p = self.config['prediction_len']

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

        dataset = Dataset.from_list(self.samples, features=features)
        print(f"Saving HF Dataset with {len(self.samples)} samples to {path}...")
        dataset.save_to_disk(path)
        print("Done.")
