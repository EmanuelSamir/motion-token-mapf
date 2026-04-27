import torch
import torch.nn as nn
import lightning as L
import numpy as np
import os
import matplotlib.pyplot as plt
from src.models.motion_lm import MotionLM
from src.utils.tokenizer import MotionTokenizer


class MotionLMLightningModule(L.LightningModule):
    """
    Simplified PyTorch Lightning Module for MotionLM.
    Handles training, validation, and visualization for highway trajectories.
    """

    def __init__(
        self,
        model_config: dict,
        lr: float = 0.0006,
        min_lr: float = 0.0001,
        weight_decay: float = 0.05,
        viz_dir: str = "visualizations",
        loss_type: str = "ce",  # "ce", "focal", or "spatial_ce"
        loss_gamma: float = 5.0,
        smoothing_sigma: float = 0.5,
        history_noise_std: float = 0.05, # Noise in meters/vel for robustness
        loss_alpha: float = 0.5,     # Weight for constant velocity token
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MotionLM(**model_config)
        self.tokenizer = MotionTokenizer()
        
        # Create weighted criterion to handle token 84 dominance
        weights = torch.ones(self.tokenizer.vocab_size + 1)
        if hasattr(self.hparams, "loss_alpha"):
            weights[84] = self.hparams.loss_alpha
        self.register_buffer("class_weights", weights)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.tokenizer.INVALID_TOKEN
        )

        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir, exist_ok=True)

        if self.hparams.loss_type == "spatial_ce":
            self._precompute_spatial_smoothing_matrix()

    def forward(self, history, tokens, agent_ids, time_ids):
        return self.model(history, tokens, agent_ids, time_ids)

    def autoregressive_rollout(self, history):
        """
        Full joint autoregressive rollout across all agents and time steps.
        Args:
            history: [Batch, N, T_hist, D_feat]
        Returns:
            all_tokens: [Batch, N * T_pred] integer tokens
        """
        B, N, T_hist, D = history.shape
        T_pred = self.hparams.model_config["max_timesteps"]
        
        # SOS Initialization
        curr_tokens = torch.full((B, 1), fill_value=84, device=self.device)
        curr_agent_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        curr_time_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        
        all_rollout_tokens = []
        
        mem = self.model.encoder(history)
        
        for t in range(T_pred):
            for n_idx in range(N):
                # Predict next token
                logits = self.model.decoder(curr_tokens, mem, curr_agent_ids, curr_time_ids)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                
                all_rollout_tokens.append(next_token)
                
                if len(all_rollout_tokens) < N * T_pred:
                    # Update sequence for next prediction
                    curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
                    
                    next_n = (n_idx + 1) % N
                    next_t = t + 1 if next_n == 0 else t
                    
                    n_id = torch.full((B, 1), fill_value=next_n, dtype=torch.long, device=self.device)
                    t_id = torch.full((B, 1), fill_value=next_t, dtype=torch.long, device=self.device)
                    curr_agent_ids = torch.cat([curr_agent_ids, n_id], dim=1)
                    curr_time_ids = torch.cat([curr_time_ids, t_id], dim=1)
        
        return torch.cat(all_rollout_tokens, dim=1) # [B, N*T]

    def training_step(self, batch, batch_idx):
        history = batch["history"]
        tokens = batch["tokens"]
        agent_ids = batch["agent_ids"]
        time_ids = batch["time_ids"]

        # 1. Noise Injection (Only during training for robustness)
        if self.hparams.history_noise_std > 0:
            noise = torch.randn_like(history) * self.hparams.history_noise_std
            # Apply only to x, y, vx, vy (indices 0:4), not heading or padding
            history = history.clone()
            history[:, :, :, :4] += noise[:, :, :, :4]

        # 2. Shift tokens for autoregressive teaching (input is t-1)
        # Use token 84 (Idle/Constant Velocity) as the SOS token instead of 0 (Panic Brake)
        input_tokens = torch.full_like(tokens, fill_value=84)
        input_tokens[:, 1:] = tokens[:, :-1]

        logits = self(history, input_tokens, agent_ids, time_ids)

        # Calculate loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = tokens.reshape(-1)

        if self.hparams.loss_type == "focal":
            loss = self.focal_loss(
                logits_flat, targets_flat, gamma=self.hparams.loss_gamma
            )
        elif self.hparams.loss_type == "spatial_ce":
            loss = self.spatial_smoothing_loss(
                logits, tokens, 
                gamma=self.hparams.loss_gamma,
                alpha=self.hparams.loss_alpha
            )
        else:
            loss = self.criterion(logits_flat, targets_flat)

        # Calculate Accuracy (ignoring padding)
        preds = logits.argmax(dim=-1)
        correct = (preds == tokens).float()
        mask = (tokens != self.tokenizer.INVALID_TOKEN).float()
        acc = (correct * mask).sum() / (mask.sum() + 1e-6)

        # Logging
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )
        self.log(
            "train/acc",
            acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=tokens.size(0),
        )

        # Log learning rate
        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], on_step=True, prog_bar=False)

        # Periodic weight movement logging (respecting log_every_n_steps)
        if self.global_step % self.trainer.log_every_n_steps == 0:
            for name, param in self.model.named_parameters():
                if "weight" in name and param.requires_grad:
                    # Log weight standard deviation (if it's 0, the layer is dead)
                    self.log(f"debug/weights_std/{name}", param.std(), on_step=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        # Only compute expensive gradient norms if it's time to log
        if self.global_step % self.trainer.log_every_n_steps == 0:
            total_grad_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm**0.5
            self.log("debug/total_grad_norm", total_grad_norm, on_step=True)

    def validation_step(self, batch, batch_idx):
        history = batch["history"]
        tokens = batch["tokens"]
        agent_ids = batch["agent_ids"]
        time_ids = batch["time_ids"]

        input_tokens = torch.full_like(tokens, fill_value=84)
        input_tokens[:, 1:] = tokens[:, :-1]

        logits = self(history, input_tokens, agent_ids, time_ids)

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = tokens.reshape(-1)

        if self.hparams.loss_type == "focal":
            loss = self.focal_loss(
                logits_flat, targets_flat, gamma=self.hparams.loss_gamma
            )
        elif self.hparams.loss_type == "spatial_ce":
            loss = self.spatial_smoothing_loss(
                logits, tokens, 
                gamma=self.hparams.loss_gamma,
                alpha=self.hparams.loss_alpha
            )
        else:
            loss = self.criterion(logits_flat, targets_flat)

        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == tokens).float()
        mask = (tokens != self.tokenizer.INVALID_TOKEN).float()
        acc = (correct * mask).sum() / (mask.sum() + 1e-6)

        self.log(
            "val/loss", loss, prog_bar=True, on_epoch=True, batch_size=tokens.size(0)
        )
        self.log(
            "val/acc", acc, prog_bar=True, on_epoch=True, batch_size=tokens.size(0)
        )

        # ADE / FDE Calculation (Physical Metrics)
        ade_list, fde_list = [], []
        t_gt_coords = batch["gt_future"].cpu().numpy()  # [B, N, T+1, 2]
        m_mask = batch["m_mask"].cpu().numpy()  # [B, N, T+1]
        init_deltas = batch["initial_deltas"].cpu().numpy() # [B, N, 2]
        
        preds_all = preds.cpu().numpy() # [B, L] 
        B, N_max = t_gt_coords.shape[0], t_gt_coords.shape[1]
        L = preds_all.shape[1]
        T_actual = L // N_max

        for b in range(B):
            for n in range(N_max):
                if m_mask[b, n].sum() < 2: continue # Pad agent
                
                # Extract predicted tokens for this specific agent (Time-First Interleaving)
                agent_tokens = [preds_all[b, tt * N_max + n] for tt in range(T_actual)]
                
                # Reconstruct
                start_pos = t_gt_coords[b, n, 0]
                traj_pred = self.tokenizer.reconstruct_trajectory(
                    start_pos, agent_tokens, initial_delta=init_deltas[b, n]
                )
                
                # Compare with GT
                active_steps = int(m_mask[b, n].sum())
                traj_gt = t_gt_coords[b, n][:active_steps]
                
                # Use min length to avoid shape mismatch if pred stopped early
                min_len = min(len(traj_pred), len(traj_gt))
                if min_len < 2: continue 
                
                # Metrics (Standard ADE: Average of Euclidean Distances)
                distances = np.sqrt(np.sum((traj_pred[:min_len] - traj_gt[:min_len])**2, axis=1))
                ade_list.append(np.mean(distances))
                fde_list.append(distances[-1])

        # --- 2. Autoregressive Rollout (The Real Test) ---
        rollout_ade_list = []
        if batch_idx == 0:
            rollout_tokens = self.autoregressive_rollout(history)
            rollout_tokens_np = rollout_tokens.cpu().numpy()
            
            for b in range(B):
                for n in range(N_max):
                    if m_mask[b, n].sum() < 2: continue
                    
                    agent_tokens = [rollout_tokens_np[b, tt * N_max + n] for tt in range(T_actual)]
                    traj_rollout = self.tokenizer.reconstruct_trajectory(
                        t_gt_coords[b, n, 0], agent_tokens, initial_delta=init_deltas[b, n]
                    )
                    
                    min_len = min(len(traj_rollout), len(t_gt_coords[b, n][m_mask[b, n] > 0]))
                    if min_len < 2: continue
                    
                    dist = np.linalg.norm(traj_rollout[:min_len] - t_gt_coords[b, n][:min_len], axis=1)
                    rollout_ade_list.append(np.mean(dist))

        # --- 3. Logging ---
        if ade_list:
            self.log("val/ADE", np.mean(ade_list), prog_bar=True, on_epoch=True)
            self.log("val/FDE", np.mean(fde_list), on_epoch=True)
        if rollout_ade_list:
            self.log("val/rollout_ADE", np.mean(rollout_ade_list), prog_bar=True, on_epoch=True)

        # Visualization check (Randomly pick ~5% of batches per validation cycle)
        if self.trainer.sanity_checking:
            return loss

        import random

        if random.random() < 0.025:
            self.visualize_prediction(
                batch, batch_idx, prefix=f"val_ep{self.current_epoch}"
            )
            self.log_token_diagnostics(logits, tokens)

        return loss

    def _precompute_spatial_smoothing_matrix(self):
        """
        Precomputes a [169, 169] matrix where S[i, j] is the distance-based
        target probability assigned to token j when the ground truth is token i.
        """
        num_bins = self.tokenizer.num_bins
        total_tokens = self.tokenizer.vocab_size  # 169
        sigma = self.hparams.smoothing_sigma

        # Grid coordinates for every token
        indices = torch.arange(total_tokens)
        ix = indices // num_bins
        iy = indices % num_bins

        # Calculate pairwise Euclidean distances in the grid
        diff_x = ix.view(-1, 1) - ix.view(1, -1)
        diff_y = iy.view(-1, 1) - iy.view(1, -1)
        distances_sq = diff_x**2 + diff_y**2

        # Gaussian smoothing
        smoothing_matrix = torch.exp(-distances_sq / (2 * sigma**2))

        # Normalize rows so they sum to 1.0 (valid probability distribution)
        smoothing_matrix = smoothing_matrix / smoothing_matrix.sum(dim=1, keepdim=True)
        self.register_buffer("spatial_smoothing_matrix", smoothing_matrix)

    def spatial_smoothing_loss(self, logits, targets, gamma=5.0, alpha=0.5):
        """
        Hyper-Aggressive Spatial Focal Loss.
        - gamma=5.0: Heavily suppresses easy examples (Gamma).
        - alpha=0.5: Modestly de-weights the dominant 'constant velocity' token (Alpha).
        """
        import torch.nn.functional as F
        
        # 1. Flatten time and batch
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        
        # 2. Mask valid targets (ignoring padding)
        valid_mask = targets != self.tokenizer.INVALID_TOKEN
        if not valid_mask.any():
            return logits.sum() * 0.0
            
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        
        # 3. Get soft targets from the precomputed matrix [num_valid, 169]
        soft_targets = self.spatial_smoothing_matrix[valid_targets]
        
        # 4. Compute Log Probabilities (sliced to size of smoothing matrix)
        log_probs = F.log_softmax(valid_logits, dim=-1)[:, :169]
        probs = torch.exp(log_probs)
        
        # 5. Compute Cross Entropy against soft targets
        ce_per_sample = -(soft_targets * log_probs).sum(dim=-1)
        
        # 6. Apply Focal Weighting (Gamma)
        # p_weighted: Integrated probability in the 'correct area'
        p_weighted = (soft_targets * probs).sum(dim=-1)
        focal_weight = (1 - p_weighted) ** gamma
        
        # 7. Apply Alpha Balancing (Specifically for token 84)
        alpha_weight = torch.ones_like(ce_per_sample)
        alpha_weight[valid_targets == 84] = alpha
        
        return (focal_weight * alpha_weight * ce_per_sample).mean()

    def focal_loss(self, logits, targets, gamma=2.0):
        """
        Standard Focal Loss implementation to handle imbalanced datasets.
        Down-weights easy examples and focuses on hard maneuvers.
        """
        import torch.nn.functional as F

        ce_loss = F.cross_entropy(
            logits, targets, ignore_index=self.tokenizer.INVALID_TOKEN, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        f_loss = (1 - pt) ** gamma * ce_loss
        return f_loss.mean()

    def visualize_prediction(self, batch, batch_idx, prefix="val"):
        """
        Performs fast Teacher Forcing inference (Training-style) and saves plots.
        This is 200x faster than autoregressive sampling.
        """
        self.model.eval()
        device = self.device

        # Data extraction
        history = batch["history"].to(device)  # [B, N, T_hist, 5]
        tokens = batch["tokens"].to(device)  # [B, N*T]
        agent_ids = batch["agent_ids"].to(device)
        time_ids = batch["time_ids"].to(device)
        t_gt_coords = batch["gt_future"].cpu().numpy()  # [B, N, T_pred+1, 2]
        m_mask = batch["m_mask"].cpu().numpy()  # [B, N, T_pred+1]
        init_deltas = batch["initial_deltas"].cpu().numpy()  # [B, N, 2]

        N = self.hparams.model_config["max_agents"]
        T = self.hparams.model_config["max_timesteps"]

        # Full Autoregressive Rollout for the first sample
        with torch.no_grad():
            rollout_tokens = self.autoregressive_rollout(history[:1]) # [1, N*T]
            pred_tokens_all = rollout_tokens.cpu().numpy()

        L = pred_tokens_all.shape[1]
        T_actual = L // N

        # Plot only the first sample of the batch for efficiency
        fig, ax = plt.subplots(figsize=(15, 6))
        colors = plt.cm.get_cmap("tab10", N)

        for n in range(N):
            if m_mask[0, n].sum() == 0:
                continue  # Skip empty padding agents

            color = colors(n)
            hist_n = history[0, n].cpu().numpy()

            # Ground Truth
            active_mask = m_mask[0, n] > 0
            traj_gt = t_gt_coords[0, n][active_mask]
            ax.plot(
                traj_gt[:, 0],
                traj_gt[:, 1],
                color=color,
                marker="D",
                markersize=3,
                linestyle="-",
                linewidth=1,
                alpha=0.2,
                label=f"Agent {n} GT" if n < 5 else "",
            )

            # Prediction (AUTOREGRESSIVE ROLLOUT)
            # We extract the interleaved tokens for this agent (Time-First)
            agent_pred_tokens = [
                pred_tokens_all[0, tt * N + n] for tt in range(T_actual)
            ]
            
            # Reconstruct
            start_pos = t_gt_coords[0, n, 0]
            traj_pred = self.tokenizer.reconstruct_trajectory(
                start_pos, agent_pred_tokens, initial_delta=init_deltas[0, n]
            )
            
            ax.plot(
                traj_pred[:, 0],
                traj_pred[:, 1],
                color=color,
                marker="o",
                markersize=4,
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Agent {n} Rollout" if n < 5 else "",
            )

            valid_steps = min(len(traj_pred), int(active_mask.sum()))
            
            # Calculate ADE/FDE for this agent
            distances = np.linalg.norm(traj_pred[:valid_steps] - traj_gt[:valid_steps], axis=1)
            ade_n = distances.mean()
            fde_n = distances[-1]

            ax.plot(
                traj_pred[:valid_steps, 0],
                traj_pred[:valid_steps, 1],
                color=color,
                marker="x",
                markersize=5,
                linestyle="--",
                alpha=0.9,
                label=f"A{n} Pred (Acc:{agent_acc:.0f}%, FDE:{fde_n:.1f}m)" if n < 5 else "",
            )

        ax.axhline(
            y=10.0, color="gray", linestyle="--", alpha=0.3, label="Ramp Boundary"
        )
        ax.set_title(f"Visual Debug | Step {self.global_step} | Batch {batch_idx}")
        ax.set_xlabel("Global X (meters)")
        ax.set_ylabel("Global Y (meters)")
        ax.grid(True, linestyle=":", alpha=0.3)

        # Position legend at the bottom
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=4,
            fontsize="x-small",
        )

        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

        save_path = os.path.join(
            self.hparams.viz_dir,
            f"{prefix}_step{self.global_step}_batch{batch_idx}.png",
        )
        fig.savefig(save_path)
        plt.close(fig)

    def log_token_diagnostics(self, logits, targets):
        """
        Generates 2D heatmaps of predicted vs ground truth tokens (13x13 grid).
        Logged to Tensorboard as images.
        """
        preds = logits.argmax(dim=-1).detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        
        # Filter out invalid tokens
        mask = targets != self.tokenizer.INVALID_TOKEN
        preds = preds[mask]
        targets = targets[mask]
        
        if len(targets) == 0: return

        # Dimensions
        num_bins = self.tokenizer.num_bins # 13
        
        # 1. Confusion Grid (True Positives Distribution)
        tp_mask = (preds == targets)
        tp_tokens = targets[tp_mask]
        
        tp_grid = np.zeros((num_bins, num_bins))
        for t in tp_tokens:
            if t < num_bins * num_bins:
                ix, iy = t // num_bins, t % num_bins
                tp_grid[ix, iy] += 1
        
        # 2. Error Grid (Where do we fail?)
        err_mask = (preds != targets)
        err_tokens = preds[err_mask] # Where the model predicted wrongly
        
        err_grid = np.zeros((num_bins, num_bins))
        for t in err_tokens:
            if t < num_bins * num_bins:
                ix, iy = t // num_bins, t % num_bins
                err_grid[ix, iy] += 1

        # Normalize for visualization
        tp_grid = tp_grid / (tp_grid.max() + 1e-6)
        err_grid = err_grid / (err_grid.max() + 1e-6)

        # Apply Log Scale for visibility
        tp_map_log = np.log10(tp_grid + 1)
        err_map_log = np.log10(err_grid + 1)
        
        # Create Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.imshow(tp_map_log, cmap='viridis', origin='lower')
        ax1.set_title("True Positives (Log Scale)")
        ax1.set_xlabel("Delta-Delta Y Index")
        ax1.set_ylabel("Delta-Delta X Index")
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(err_map_log, cmap='magma', origin='lower')
        ax2.set_title("Prediction Errors (Log Scale)")
        ax2.set_xlabel("Delta-Delta Y Index")
        ax2.set_ylabel("Delta-Delta X Index")
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        # Log to Tensorboard
        self.logger.experiment.add_figure(
            "diagnostics/token_heatmaps", fig, global_step=self.global_step
        )
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine Annealing with a floor (min_lr)
        # We use T_max=100 as a default based on typical training runs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=self.hparams.min_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
