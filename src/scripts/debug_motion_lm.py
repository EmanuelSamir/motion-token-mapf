import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models.motion_lm import MotionLM
from src.utils.tokenizer import MotionTokenizer

class SyntheticInteractionDataset(Dataset):
    def __init__(self, num_samples=1000, history_len=10, prediction_len=20, tokenizer=None):
        self.num_samples = num_samples
        self.history_len = history_len
        self.prediction_len = prediction_len
        self.tokenizer = tokenizer or MotionTokenizer()
        self.num_agents = 3
        self.data = self._generate_data()

    def _generate_data(self):
        samples = []
        for i in range(self.num_samples):
            scenario = i % 4
            T = self.history_len + self.prediction_len
            trajs = np.zeros((self.num_agents, T, 5))
            v0, dt = 10.0, 0.2
            
            # Init Positions
            trajs[0, 0] = [40, 0, v0, 0, 0]; trajs[1, 0] = [20, 0, v0, 0, 0]; trajs[2, 0] = [0, 0, v0, 0, 0]
            if scenario == 2: trajs[0, 0, 1] = 4.0 # Cut-in starts in other lane
            
            for n in range(self.num_agents):
                curr_v, curr_y = v0, trajs[n, 0, 1]
                for t in range(1, T):
                    acc, vy = 0.0, 0.0
                    
                    # SIGNATURES MUST START IN HISTORY (t < 10)
                    t_start = 5 
                    
                    if scenario == 1: # Brake
                        if n == 0 and t >= t_start: acc = -5.0
                        if n == 1 and t >= t_start + 4: acc = -5.0
                    elif scenario == 2: # Cut-in
                        if n == 0 and t >= t_start:
                            if curr_y > 0.05: vy = -2.5
                        if n == 1 and t >= t_start + 6 and trajs[0, t, 1] < 1.0: acc = -3.0
                    elif scenario == 3: # Swerve
                        ts = t_start + n*2
                        if t >= ts and t < ts + 5: vy = 3.0
                        elif t >= ts + 5 and t < ts + 10: vy = -3.0

                    curr_v = max(0, curr_v + acc * dt)
                    curr_y = curr_y + vy * dt
                    trajs[n, t] = [trajs[n, t-1, 0] + curr_v * dt, curr_y, curr_v, vy, 0]
            
            hists, tokens, init_deltas = [], [], []
            for n in range(self.num_agents):
                h = trajs[n, :self.history_len].copy()
                f = trajs[n, self.history_len-1:, :2].copy()
                ref = h[-1, :2].copy()
                # Use recorded velocity [vx, vy] * dt directly
                idelta = h[-1, 2:4] * 0.2
                h[:, :2] -= ref; f -= ref
                hists.append(h); tokens.append(self.tokenizer.tokenize_trajectory(f, initial_delta=idelta))
                init_deltas.append(idelta)
                
            interleaved = [tokens[n][t] for t in range(self.prediction_len) for n in range(self.num_agents)]
            samples.append({
                'history': torch.tensor(np.stack(hists), dtype=torch.float32),
                'tokens': torch.tensor(interleaved, dtype=torch.long),
                'init_deltas': torch.tensor(np.stack(init_deltas), dtype=torch.float32),
                'scenario': scenario,
                'gt_future_raw': trajs[:, self.history_len-1:, :2]
            })
        return samples

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        s = self.data[idx]
        return {'history': s['history'], 'tokens': s['tokens'],
                'agent_ids': torch.tensor(np.tile(np.arange(3), 20), dtype=torch.long),
                'time_ids': torch.tensor(np.repeat(np.arange(20), 3), dtype=torch.long),
                'init_deltas': s['init_deltas'], 'scenario': s['scenario']}

def evaluate_and_viz(model, dataset, tokenizer, device):
    print("📊 Evaluating Distinct Scenarios (ADE/FDE)...")
    model.eval()
    scenarios = ["1: Constant", "2: Braking", "3: Cut-in", "4: Swerve"]
    ade_sums = [0]*4; fde_sums = [0]*4; counts = [0]*4
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    N, T, START_TOKEN = 3, 20, 84

    for idx in range(min(100, len(dataset))): # More eval samples for stability
        s = dataset.data[idx]
        h = s['history'].unsqueeze(0).to(device)
        ideltas = s['init_deltas']
        sc = s['scenario']
        
        with torch.no_grad():
            gen_tokens = torch.full((1, N*T), START_TOKEN, dtype=torch.long, device=device)
            a_ids = torch.arange(N).repeat(T).unsqueeze(0).to(device)
            tm_ids = torch.arange(T).repeat_interleave(N).unsqueeze(0).to(device)
            preds = []
            for i in range(N*T):
                logits = model(h, gen_tokens, a_ids, tm_ids)
                next_token = logits[0, i].argmax()
                if i < N*T - 1: gen_tokens[0, i+1] = next_token
                preds.append(next_token.item())

        for n in range(N):
            y_off = n * 5.0
            agent_tokens_pred = [preds[t*N + n] for t in range(T)]
            traj_pred = tokenizer.reconstruct_trajectory([0, 0], agent_tokens_pred, initial_delta=ideltas[n])
            traj_gt = s['gt_future_raw'][n] - s['gt_future_raw'][n, 0]
            errs = np.linalg.norm(traj_pred - traj_gt, axis=1)
            ade_sums[sc] += np.mean(errs); fde_sums[sc] += errs[-1]; counts[sc] += 1
            if idx < 4:
                ax = axes[idx]
                ax.plot(traj_gt[:, 0], traj_gt[:, 1] + y_off, color='gray', linestyle='--', alpha=0.3)
                ax.plot(traj_pred[:, 0], traj_pred[:, 1] + y_off, marker='x', markersize=3, label=f'A{n}' if idx==0 else "")

    print("\n" + "="*45); print(f"{'Scenario':<15} | {'ADE (m)':<10} | {'FDE (m)':<10}"); print("-" * 45)
    for i in range(4): print(f"{scenarios[i]:<15} | {ade_sums[i]/max(1, counts[i]):<10.3f} | {fde_sums[i]/max(1, counts[i]):<10.3f}")
    axes[0].legend(); plt.tight_layout(); plt.savefig("debug_result.png"); plt.close()

def train_debug():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = MotionTokenizer()
    dataset = SyntheticInteractionDataset(num_samples=1200, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MotionLM(max_agents=3, max_timesteps=20, vocab_size=170).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Stable LR
    criterion = nn.CrossEntropyLoss()
    START_TOKEN = 84
    
    print(f"🌀 Training for 60 epochs (Unique Signature Data)... Device: {device}")
    for epoch in range(60):
        epoch_loss = 0
        for batch in dataloader:
            h, tokens = batch['history'].to(device), batch['tokens'].to(device)
            a_ids, tm_ids = batch['agent_ids'].to(device), batch['time_ids'].to(device)
            in_tokens = torch.full_like(tokens, START_TOKEN)
            in_tokens[:, 1:] = tokens[:, :-1]
            logits = model(h, in_tokens, a_ids, tm_ids)
            loss = criterion(logits.reshape(-1, 170), tokens.reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); epoch_loss += loss.item()
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")

    evaluate_and_viz(model, dataset, tokenizer, device)

if __name__ == "__main__":
    train_debug()
