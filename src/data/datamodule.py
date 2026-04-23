import lightning as L
from torch.utils.data import DataLoader, random_split
from src.data.dataset import TrajectoryDataset
from src.data.hf_dataset import HFTrajectoryDataset
from src.utils.tokenizer import MotionTokenizer

class HighwayTrajectoryDataModule(L.LightningDataModule):
    """
    LightningDataModule for loading highway trajectory data.
    """
    def __init__(
        self,
        data_dir: str = "data/trajectories",
        batch_size: int = 32,
        max_agents_per_scene: int = 10,
        history_len: int = 10,
        prediction_len: int = 25,
        stride: int = 5,
        neighbor_radius: float = 50.0,
        stationary_keep_ratio: float = 0.1,
        train_val_split: float = 0.9,
        num_workers: int = 4,
        sample_ratio: float = 1.0,
        use_hf: bool = False,
        hf_path: str = "data/trajectories/hdv_1000_ep_hf"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = MotionTokenizer()

    def setup(self, stage: str = None):
        if self.hparams.use_hf:
            print(f"--- Loading Hugging Face Dataset from: {self.hparams.hf_path} (Ratio: {self.hparams.sample_ratio}) ---")
            dataset = HFTrajectoryDataset(
                path=self.hparams.hf_path, 
                sample_ratio=self.hparams.sample_ratio
            )
        else:
            print(f"--- Loading Legacy Pickle Dataset from: {self.hparams.data_dir} ---")
            dataset = TrajectoryDataset(
                data_dir=self.hparams.data_dir,
                history_len=self.hparams.history_len,
                prediction_len=self.hparams.prediction_len,
                neighbor_radius=self.hparams.neighbor_radius,
                tokenizer=self.tokenizer,
                stride=self.hparams.stride,
                sample_ratio=self.hparams.sample_ratio,
                stationary_keep_ratio=self.hparams.stationary_keep_ratio,
                max_agents_per_scene=self.hparams.max_agents_per_scene
            )
        
        train_size = int(self.hparams.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()
