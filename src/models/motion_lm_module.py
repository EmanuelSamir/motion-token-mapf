import torch
import torch.nn as nn
import lightning as L
import numpy as np
import os
import matplotlib.pyplot as plt
from src.models.motion_lm import MotionLM
from src.utils.tokenizer import MotionTokenizer
import random


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
        self.tokenizer = MotionTokenizer()
        
        # Update model config with correct vocab size (+1 for INVALID_TOKEN)
        model_config["vocab_size"] = self.tokenizer.vocab_size + 1
        self.model = MotionLM(**model_config)
        
        # Dynamically find the SOS/Idle token (constant velocity [0,0] offset)
        self.idle_token = self.tokenizer.INVALID_TOKEN
        for i, off in enumerate(self.tokenizer.codebook):
            if off[0] == 0 and off[1] == 0:
                self.idle_token = i
                break

        # Create weighted criterion
        # Model output is vocab_size + 1 (for padding), so weights must match
        weights = torch.ones(self.tokenizer.vocab_size + 1)
        if hasattr(self.hparams, "loss_alpha") and self.idle_token < self.tokenizer.vocab_size:
            weights[self.idle_token] = self.hparams.loss_alpha
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
        curr_tokens = torch.full((B, 1), fill_value=self.idle_token, device=self.device)
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

        # 1. Noise Injection
        if self.hparams.history_noise_std > 0:
            noise = torch.randn_like(history) * self.hparams.history_noise_std
            history = history.clone()
            history[:, :, :, :4] += noise[:, :, :, :4]

        # 2. Shift tokens for autoregressive teaching
        input_tokens = torch.full_like(tokens, fill_value=self.idle_token)
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

        # Calculate Accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == tokens).float()
        mask = (tokens != self.tokenizer.INVALID_TOKEN).float()
        acc = (correct * mask).sum() / (mask.sum() + 1e-6)

        # Logging
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=tokens.size(0))
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=tokens.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        history = batch["history"]
        tokens = batch["tokens"]
        agent_ids = batch["agent_ids"]
        time_ids = batch["time_ids"]

        input_tokens = torch.full_like(tokens, fill_value=self.idle_token)
        input_tokens[:, 1:] = tokens[:, :-1]

        logits = self(history, input_tokens, agent_ids, time_ids)

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = tokens.reshape(-1)

        if self.hparams.loss_type == "focal":
            loss = self.focal_loss(logits_flat, targets_flat, gamma=self.hparams.loss_gamma)
        elif self.hparams.loss_type == "spatial_ce":
            loss = self.spatial_smoothing_loss(logits, tokens, gamma=self.hparams.loss_gamma, alpha=self.hparams.loss_alpha)
        else:
            loss = self.criterion(logits_flat, targets_flat)

        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == tokens).float()
        mask = (tokens != self.tokenizer.INVALID_TOKEN).float()
        acc = (correct * mask).sum() / (mask.sum() + 1e-6)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=tokens.size(0))
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, batch_size=tokens.size(0))

        # ADE / FDE
        ade_list, fde_list = [], []
        t_gt_coords = batch["gt_future"].cpu().numpy()
        m_mask = batch["m_mask"].cpu().numpy()
        init_deltas = batch["initial_deltas"].cpu().numpy()
        
        preds_all = preds.cpu().numpy()
        B, N_max = t_gt_coords.shape[0], t_gt_coords.shape[1]
        L = preds_all.shape[1]
        T_actual = L // N_max

        for b in range(B):
            for n in range(N_max):
                if m_mask[b, n].sum() < 2: continue
                agent_tokens = [preds_all[b, tt * N_max + n] for tt in range(T_actual)]
                start_pos = t_gt_coords[b, n, 0]
                traj_pred = self.tokenizer.reconstruct_trajectory(start_pos, agent_tokens, initial_delta=init_deltas[b, n])
                active_steps = int(m_mask[b, n].sum())
                traj_gt = t_gt_coords[b, n][:active_steps]
                min_len = min(len(traj_pred), len(traj_gt))
                if min_len < 2: continue 
                dist = np.sqrt(np.sum((traj_pred[:min_len] - traj_gt[:min_len])**2, axis=1))
                ade_list.append(np.mean(dist))
                fde_list.append(dist[-1])

        if ade_list:
            self.log("val/ADE", np.mean(ade_list), prog_bar=True, on_epoch=True)
            self.log("val/FDE", np.mean(fde_list), on_epoch=True)

        if random.random() < 0.025:
            self.visualize_prediction(batch, batch_idx, prefix=f"val_ep{self.current_epoch}")
            self.log_token_diagnostics(logits, tokens)

        return loss

    def _precompute_spatial_smoothing_matrix(self):
        """
        Uses physical grid distance between codebook offsets.
        """
        total_tokens = self.tokenizer.vocab_size 
        sigma = self.hparams.smoothing_sigma
        coords = torch.from_numpy(self.tokenizer.codebook).float()
        diff = coords.view(-1, 1, 2) - coords.view(1, -1, 2)
        distances_sq = (diff**2).sum(dim=-1)
        smoothing_matrix = torch.exp(-distances_sq / (2 * sigma**2))
        smoothing_matrix = smoothing_matrix / smoothing_matrix.sum(dim=1, keepdim=True)
        self.register_buffer("spatial_smoothing_matrix", smoothing_matrix)

    def spatial_smoothing_loss(self, logits, targets, gamma=5.0, alpha=0.5):
        import torch.nn.functional as F
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        valid_mask = targets != self.tokenizer.INVALID_TOKEN
        if not valid_mask.any(): return logits.sum() * 0.0
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        soft_targets = self.spatial_smoothing_matrix[valid_targets]
        log_probs = F.log_softmax(valid_logits, dim=-1)[:, :self.tokenizer.vocab_size]
        probs = torch.exp(log_probs)
        ce_per_sample = -(soft_targets * log_probs).sum(dim=-1)
        p_weighted = (soft_targets * probs).sum(dim=-1)
        focal_weight = (1 - p_weighted) ** gamma
        alpha_weight = torch.ones_like(ce_per_sample)
        alpha_weight[valid_targets == self.idle_token] = alpha
        return (focal_weight * alpha_weight * ce_per_sample).mean()

    def focal_loss(self, logits, targets, gamma=2.0):
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.INVALID_TOKEN, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** gamma * ce_loss).mean()

    def visualize_prediction(self, batch, batch_idx, prefix="val"):
        self.model.eval()
        device = self.device
        history = batch["history"].to(device)
        t_gt_coords = batch["gt_future"].cpu().numpy()
        m_mask = batch["m_mask"].cpu().numpy()
        init_deltas = batch["initial_deltas"].cpu().numpy()
        N = self.hparams.model_config["max_agents"]
        with torch.no_grad():
            rollout_tokens = self.autoregressive_rollout(history[:1])
            pred_tokens_all = rollout_tokens.cpu().numpy()
        L = pred_tokens_all.shape[1]
        T_actual = L // N
        fig, ax = plt.subplots(figsize=(15, 6))
        colors = plt.cm.get_cmap("tab10", N)
        for n in range(N):
            if m_mask[0, n].sum() == 0: continue
            color = colors(n)
            active_mask = m_mask[0, n] > 0
            traj_gt = t_gt_coords[0, n][active_mask]
            ax.plot(traj_gt[:, 0], traj_gt[:, 1], color=color, marker="D", markersize=3, alpha=0.2)
            agent_pred_tokens = [pred_tokens_all[0, tt * N + n] for tt in range(T_actual)]
            traj_pred = self.tokenizer.reconstruct_trajectory(t_gt_coords[0, n, 0], agent_pred_tokens, initial_delta=init_deltas[0, n])
            ax.plot(traj_pred[:, 0], traj_pred[:, 1], color=color, linestyle="--", alpha=0.8, label=f"A{n} Rollout")
        ax.set_title(f"Visual Debug | Step {self.global_step}")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="x-small")
        ax.set_aspect("equal")
        plt.tight_layout()
        save_path = os.path.join(self.hparams.viz_dir, f"{prefix}_step{self.global_step}.png")
        fig.savefig(save_path)
        plt.close(fig)

    def log_token_diagnostics(self, logits, targets):
        preds = logits.argmax(dim=-1).detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        mask = targets != self.tokenizer.INVALID_TOKEN
        preds = preds[mask]
        targets = targets[mask]
        if len(targets) == 0: return
        codebook = self.tokenizer.codebook
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        counts = np.bincount(targets, minlength=len(codebook))
        ax1.scatter(codebook[:, 0], codebook[:, 1], c=np.log10(counts + 1), cmap='viridis', s=50)
        ax1.set_title("Target Distribution (Log Scale)")
        err_counts = np.bincount(preds[preds != targets], minlength=len(codebook))
        ax2.scatter(codebook[:, 0], codebook[:, 1], c=np.log10(err_counts + 1), cmap='magma', s=50)
        ax2.set_title("Mis-prediction Hotspots (Log Scale)")
        plt.tight_layout()
        self.logger.experiment.add_figure("diagnostics/token_codebook_heatmaps", fig, global_step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=self.hparams.min_lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

