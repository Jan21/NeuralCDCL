import matplotlib.pyplot as plt
import pickle
import torch
import os
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerFast
from hydra.utils import get_original_cwd, to_absolute_path
from typing import Optional, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Datamodule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        
        # Initialize data collator once
        self.collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def setup(self, stage=None):
        # Dynamically identify splits (train_*, val_*)
        self.train_split = [k for k in self.dataset.keys() if k.startswith("train_")][0]
        self.val_splits = [k for k in self.dataset.keys() if k.startswith("val_")]

    def train_dataloader(self):
        return DataLoader(
            self.dataset[self.train_split],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        # Return list of dataloaders (one per validation split)
        return [
            DataLoader(
                self.dataset[val_split],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val_split in self.val_splits
        ]

    def test_dataloader(self):
        # Optional: Add test splits similarly
        pass

    def connect(
        self, max_seq_length: Optional[int] = None
    ) -> None:
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

def plot_token_length_histograms(tokenized_dataset):
    """Plot histograms of token lengths for each validation dataset."""
    val_splits = [split for split in tokenized_dataset.keys() if split.startswith("val")]

    for split in val_splits:
        token_lengths = [len(seq) for seq in tokenized_dataset[split]["input_ids_unpadded"]]

        plt.figure()
        plt.hist(token_lengths, bins=30, edgecolor="black")
        plt.title(f"Token Length Distribution - {split}")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"{split}_token_length.png") # Save with split name

def get_data(cfg: DictConfig, tokenizer):
    hf_dataset = load_dataset(
        "json",
        data_files={
            f"train_{cfg.data.train_data}": to_absolute_path(
                os.path.join(cfg.data.datapath, cfg.data[f"train_file_{cfg.data.train_data}"])
            ),
            "val_easy": to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.val_file_easy)),
            "val_medium": to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.val_file_medium)),
            "val_hard": to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.val_file_hard)),
        },
    )

    # Subset data if needed
    for split in hf_dataset.keys():
        if split.startswith("train"):
            hf_dataset[split] = hf_dataset[split].select(range(cfg.data.num_train))
        elif split.startswith("val"):
            hf_dataset[split] = hf_dataset[split].select(range(cfg.data.num_eval))

    # Tokenize all splits without padding to compute actual token lengths
    def tokenize_unpadded(element):
        outputs = tokenizer(
            [tokenizer.bos_token + text.strip() + tokenizer.eos_token for text in element["text"]],
            truncation=True,
            max_length=10000,
        )
        return {"input_ids_unpadded": outputs["input_ids"]}

    # tokenized_unpadded_dataset = hf_dataset.map(tokenize_unpadded, batched=True, remove_columns=hf_dataset[f"train_{cfg.data.train_data}"].column_names)
    # plot_token_length_histograms(tokenized_unpadded_dataset)

    def tokenize_padded(element):
        outputs = tokenizer(
            [tokenizer.bos_token + text.strip() + tokenizer.eos_token for text in element["text"]],
            truncation=True,
            max_length=cfg.model.block_size,
            padding="max_length",
        )
        return {"input_ids": outputs["input_ids"]}

    tokenized_dataset = hf_dataset.map(tokenize_padded, batched=True, remove_columns=hf_dataset[f"train_{cfg.data.train_data}"].column_names)

    return tokenized_dataset

def get_tokenizer(tok_data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=to_absolute_path(tok_data.tokenizer_path))
    tokenizer.add_special_tokens({
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
    })
    return tokenizer