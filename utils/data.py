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
from datasets import Dataset, DatasetDict
import random
import numpy as np

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
        self.train_split = [k for k in self.dataset.keys() if k.startswith("train")][0]
        self.val_splits = ["val", "test"]

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
    token_lengths = [len(seq) for seq in tokenized_dataset["train"]["input_ids_unpadded"]]

    plt.figure()
    plt.hist(token_lengths, bins=30, edgecolor="black")
    plt.title(f"Token Length Distribution")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"token_length.png") # Save with split name

def weighted_random_choice(traces):
    """Selects one element from traces with increasing probability by index."""
    if not traces:
        return None  # If the list is empty, return None

    weights = np.linspace(1, 2, len(traces))  # Increasing weight distribution
    probabilities = weights / weights.sum()  # Normalize to sum to 1

    return random.choices(traces, weights=probabilities, k=1)[0]  # Pick one sample

def preprocess_dataset(dataset, packed):
    """
    Expands each input datapoint into a fixed number of training examples before `.map()` is applied.
    Selects only one sample from `unit_prop_traces` and `analyze_conflict_traces` with weighted probability.
    """
    all_data = []
    for example in dataset:
        expanded_examples = []
        
        if packed:
            # Select one random sample from each trace list with increasing probability
            unit_prop_sample = weighted_random_choice(example["unit_prop_traces"])
            analyze_conflict_sample = weighted_random_choice(example["analyze_conflict_traces"])
            
            # Construct the list including SOLVE + one sampled UP + one sampled AC
            all_texts = [example["solve_trace_packed"]]
            if unit_prop_sample:
                all_texts.append(unit_prop_sample)
            if analyze_conflict_sample:
                all_texts.append(analyze_conflict_sample)
        else:
            all_texts = [example["solve_trace_unpacked"]]

        for text in all_texts:
            expanded_examples.append({"text": text.replace("[", " [ ").replace("]", " ] ")})
        all_data.extend(expanded_examples)

    return Dataset.from_list(all_data)


def get_data(cfg: DictConfig, tokenizer):
    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.train_file)),
            "test": to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.test_file)),
        },
    )

    hf_dataset["train"] = preprocess_dataset(hf_dataset["train"].select(range(cfg.data.num_train)), cfg.data.packed)
    hf_dataset["test"] = preprocess_dataset(hf_dataset["test"].select(range(cfg.data.num_test)), cfg.data.packed)

    # Split the train set into train and validation
    split_dataset = hf_dataset["train"].train_test_split(train_size=0.9, seed=42)  # 90% train, 10% val
    hf_dataset = DatasetDict({
        "train": split_dataset["train"],
        "val": split_dataset["test"],
        "test": hf_dataset["test"],  # Keep test separate
    })

    # Tokenize all splits without padding to compute actual token lengths
    def tokenize_unpadded(element):
        outputs = tokenizer(
            [tokenizer.bos_token + text.strip() + tokenizer.eos_token for text in element["text"]],
            truncation=True,
            max_length=10000,
        )
        return {"input_ids_unpadded": outputs["input_ids"]}

    # tokenized_unpadded_dataset = hf_dataset.map(tokenize_unpadded, batched=True, remove_columns=hf_dataset[f"train"].column_names)
    # plot_token_length_histograms(tokenized_unpadded_dataset)
    # exit()

    # def tokenize_padded(element):
    #     outputs = tokenizer(
    #         [tokenizer.bos_token + text.strip() + tokenizer.eos_token for text in element["text"]],
    #         truncation=True,
    #         max_length=cfg.model.block_size,
    #         padding="longest",
    #     )
    #     return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
    def remove_duplicates(dataset):
        """Removes duplicates based on the 'text' field before tokenization."""
        seen = set()
        unique_data = []
        
        for item in dataset:
            text = item["text"]
            if text not in seen:
                seen.add(text)
                unique_data.append(item)
        
        return Dataset.from_list(unique_data)

    hf_dataset["train"] = remove_duplicates(hf_dataset["train"])
    hf_dataset["val"] = remove_duplicates(hf_dataset["val"])
    hf_dataset["test"] = remove_duplicates(hf_dataset["test"])

    def filter_long_texts(example):
        """Filters out texts that are longer than block_size when tokenized."""
        tokenized = tokenizer(
            tokenizer.bos_token + example["text"].strip() + tokenizer.eos_token,
            truncation=False,  # We don't truncate to get true length
            max_length=None,   # No max length, so we see the full tokenized length
        )
        return len(tokenized["input_ids"]) <= cfg.model.block_size  # Keep only those within limit

    # Apply filtering before tokenization
    hf_dataset["train"] = hf_dataset["train"].filter(filter_long_texts)
    hf_dataset["val"] = hf_dataset["val"].filter(filter_long_texts)
    hf_dataset["test"] = hf_dataset["test"].filter(filter_long_texts)

    def tokenize_padded(element):
        outputs = tokenizer(
            [tokenizer.bos_token + text.strip() + tokenizer.eos_token for text in element["text"]],
            truncation=True,
            max_length=cfg.model.block_size,
            # padding="longest",
            padding=False,
        )

        unk_token_id = tokenizer.unk_token_id
        for i, input_ids in enumerate(outputs["input_ids"]):
            if unk_token_id in input_ids:
                original_text = element["text"][i]  # Get the original text causing [UNK]
                print(f"[WARNING] Found [UNK] token in text: {original_text}")
                print(f"Tokenized Output: {tokenizer.tokenize(original_text)}")
                print(f"Token IDs: {input_ids}")
                print(f"Decoded Output: {tokenizer.decode(input_ids)}")
                assert False, f"Unexpected [UNK] token in input text: {original_text}"

        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    tokenized_dataset = hf_dataset.map(tokenize_padded, batched=True, remove_columns=hf_dataset[f"train"].column_names)

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