from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
import torch
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import glob
import os


@hydra.main(
    config_path="../config", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    vocab = get_vocab(cfg)
    tokenizer = get_tokenizer(vocab, cfg)


def get_tokenizer(vocab, cfg):
    vocab = {s: i for i, s in enumerate(vocab)}
    # Initialize tokenizer with complete vocabulary
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(["[BOS]", "[PAD]", "[MASK]", "[UNK]", "[EOS]"])
    # Save tokenizer
    save_path = f"{cfg.tok_data.tokenizer_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print("tokenizer saved to:", save_path)

    return tokenizer


def get_vocab(cfg: DictConfig):
    # Get all JSON files from data directory and data/generalization directory
    data_files = glob.glob("data/ac/*.json") + glob.glob("data/up/*.json") + glob.glob("data/mixed/*.json") + glob.glob("data/generalization/up/*.json") + glob.glob("data/generalization/ac/*.json") + glob.glob("data/generalization/mixed/*.json") + glob.glob("data/solve/*.json") + glob.glob("data/generalization/solve/*.json")
        
    all_data = []
    for file_path in data_files:
        with open(file_path, "r") as f:
            file_data = json.load(f)
            all_data.extend(file_data)

    data = [i["text"] for i in all_data]
    data = " ".join(data)
    vocab = set(data.split())
    print("Num of tokens:", len(vocab))
    return vocab


if __name__ == "__main__":
    main()
