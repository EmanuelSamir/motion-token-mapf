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
        focal_gamma: float = 2.0,
        smoothing_sigma: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MotionLM(**model_config)
        self.tokenizer = MotionTokenizer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.INVALID_TOKEN)

        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir, exist_ok=True)

        if self.hparams.loss_type == "spatial_ce":
            self._precompute_spatial_smoothing_matrix()

    def forward(self, history, tokens, agent_ids, time_ids):
        return self.model(history, tokens, agent_ids, time_ids)

    def training_step(self, batch, batch_idx):
        history = batch["history"]
        tokens = batch["tokens"]
        agent_ids = batch["agent_ids"]
        time_ids = batch["time_ids"]

        # Shift tokens for autoregressive teaching (input is t-1)
        # Use token 84 (Idle/Constant Velocity) as the SOS token instead of 0 (Panic Brake)
        input_tokens = torch.full_like(tokens, fill_value=84)
        input_tokens[:, 1:] = tokens[:, :-1]

        logits = self(history, input_tokens, agent_ids, time_ids)

        # Calculate loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = tokens.reshape(-1)

        if self.hparams.loss_type == "focal":
            loss = self.focal_loss(
                logits_flat, targets_flat, gamma=self.hparams.focal_gamma
            )
        elif self.hparams.loss_type == "spatial_ce":
            loss = self.spatial_smoothing_loss(logits, tokens)
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
                logits_flat, targets_flat, gamma=self.hparams.focal_gamma
            )
        elif self.hparams.loss_type == "spatial_ce":
            loss = self.spatial_smoothing_loss(logits, tokens)
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

        # Visualization check (Randomly pick ~5% of batches per validation cycle)
        if self.trainer.sanity_checking:
            return loss

        import random

        if random.random() < 0.025:
            self.visualize_prediction(
                batch, batch_idx, prefix=f"val_ep{self.current_epoch}"
            )

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

    def spatial_smoothing_loss(self, logits, targets):
        """
        Cross-entropy loss with soft spatial targets.
        """
        import torch.nn.functional as F

        # Flatten time and batch
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        # Mask valid targets (ignoring padding)
        valid_mask = targets != self.tokenizer.INVALID_TOKEN
        if not valid_mask.any():
            return logits.sum() * 0.0

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        # Get soft targets from the precomputed matrix
        # [num_valid, 169]
        soft_targets = self.spatial_smoothing_matrix[valid_targets]

        # Compute Cross Entropy (KL Divergence between soft targets and log-softmax)
        log_probs = F.log_softmax(valid_logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

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

        # Fast Forward Pass (Teacher Forcing)
        # Shift tokens like in training_step (SOS = 84)
        input_tokens = torch.full_like(tokens, fill_value=84)
        input_tokens[:, 1:] = tokens[:, :-1]

        with torch.no_grad():
            logits = self.model(history, input_tokens, agent_ids, time_ids)
            # Pick the most likely token for each position
            pred_tokens_all = logits.argmax(dim=-1)  # [B, N*T]

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

            # Prediction (Teacher Forcing - "One step ahead")
            # We extract the interleaved tokens for this agent
            agent_pred_tokens = [
                pred_tokens_all[0, tt * N + n].item() for tt in range(T)
            ]
            agent_gt_tokens = [tokens[0, tt * N + n].item() for tt in range(T)]

            # Calculate per-agent accuracy
            correct_tokens = sum(
                1
                for p, g in zip(agent_pred_tokens, agent_gt_tokens)
                if p == g and g != self.tokenizer.INVALID_TOKEN
            )
            total_tokens = sum(
                1 for g in agent_gt_tokens if g != self.tokenizer.INVALID_TOKEN
            )
            agent_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0

            # Reconstruct starting from physical history
            start_pos = hist_n[-1, :2]
            traj_pred = self.tokenizer.reconstruct_trajectory(
                start_pos, agent_pred_tokens, initial_delta=init_deltas[0, n]
            )

            valid_steps = min(len(traj_pred), int(active_mask.sum()))
            ax.plot(
                traj_pred[:valid_steps, 0],
                traj_pred[:valid_steps, 1],
                color=color,
                marker="x",
                markersize=5,
                linestyle="--",
                alpha=0.9,
                label=f"A{n} Pred ({agent_acc:.0f}%)" if n < 5 else "",
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
