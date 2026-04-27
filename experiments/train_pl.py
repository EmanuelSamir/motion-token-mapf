import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import os

from src.data.datamodule import HighwayTrajectoryDataModule
from src.models.motion_lm_module import MotionLMLightningModule

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Print config
    # print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed)

    # 1. Init DataModule
    datamodule = HighwayTrajectoryDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        max_agents_per_scene=cfg.model.max_agents,
        history_len=cfg.model.get("history_len", 10),
        prediction_len=cfg.model.max_timesteps,
        num_workers=cfg.trainer.get("num_workers", 4),
        sample_ratio=cfg.data.get("sample_ratio", 1.0),
        use_hf=cfg.data.get("use_hf", False),
        hf_path=cfg.data.get("hf_path", "data/trajectories/hdv_1000_ep_hf")
    )

    # 2. Init LightningModule
    model_config = {
        "hidden_size": cfg.model.hidden_size,
        "num_encoder_layers": cfg.model.num_encoder_layers,
        "num_decoder_layers": cfg.model.num_decoder_layers,
        "num_heads": cfg.model.num_heads,
        "ff_size": cfg.model.ff_size,
        "max_agents": cfg.model.max_agents,
        "max_timesteps": cfg.model.max_timesteps,
        "dropout": cfg.model.get("dropout", 0.1)
    }
    
    model = MotionLMLightningModule(
        model_config=model_config,
        lr=cfg.model.get("lr", 0.0006),
        min_lr=cfg.model.get("min_lr", 0.0001),
        weight_decay=cfg.model.get("weight_decay", 0.6),
        viz_dir=cfg.get("viz_dir", "visualizations"),
        loss_type=cfg.model.get("loss_type", "ce"),
        loss_gamma=cfg.model.get("loss_gamma", 5.0),
        loss_alpha=cfg.model.get("loss_alpha", 0.5),
        smoothing_sigma=cfg.model.get("smoothing_sigma", 0.5),
        history_noise_std=cfg.model.get("history_noise_std", 0.05)
    )

    # 3. Init Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints",
        filename="motion-lm-{epoch:02d}-{val/loss:.4f}",
        save_top_k=3,
        mode="min"
    )

    # Check for accelerator
    accelerator = "auto"
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=cfg.trainer.get("max_epochs", 100),
        max_steps=cfg.trainer.get("max_steps", -1),
        val_check_interval=cfg.trainer.get("val_check_interval", 1000),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        callbacks=[checkpoint_callback],
        limit_val_batches=cfg.trainer.get("limit_val_batches", 10)
    )

    # 4. Train
    resume_path = cfg.trainer.get("resume_from_checkpoint")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)

if __name__ == "__main__":
    # Import torch here to avoid issues with hydra spawns
    import torch
    train()
