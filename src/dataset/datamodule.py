from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional


class Datamodule(LightningDataModule):
    def __init__(self, dataloaders: dict[str, DataLoader]):
        super().__init__()
        self.dataloaders = dataloaders

    def train_dataloader(self) -> DataLoader:
        if "train" not in self.dataloaders:
            raise ValueError("Missing 'train' dataloader -> required for training.")
        return self.dataloaders["train"]

    def val_dataloader(self) -> Optional[DataLoader]:
        return self.dataloaders.get("val", None)

    def test_dataloader(self) -> Optional[DataLoader]:
        return self.dataloaders.get("test", None)

    def ood_dataloader(self) -> Optional[DataLoader]:
        return self.dataloaders.get("ood", None)
