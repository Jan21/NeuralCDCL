from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from datasets import Dataset,DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from hydra.utils import to_absolute_path
import torch
import json
import hydra
import os
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    vocab = get_vocab(cfg)
    tokenizer = get_tokenizer(vocab, cfg)

def get_tokenizer(vocab, cfg):
    vocab = {s:i for i,s in enumerate(vocab.union({'[UNK]'}))}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer=WhitespaceSplit()
    tokenizer.add_special_tokens(['[BOS]', '[PAD]','[MASK]','[UNK]', '[EOS]'])
    tokenizer_path = to_absolute_path(cfg.tok_data.tokenizer_path)
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tokenizer.save(tokenizer_path)
    print("tokenizer saved to:", tokenizer_path)
    return tokenizer

def get_vocab(cfg: DictConfig):
    with open (cfg.data.datapath + "/" + cfg.data.train_file, "rb") as f:
        train = json.load(f)

    data = train
    data = [i["text"] for i in data]
    data = " ".join(data)
    vocab = set(data.split())
    print(vocab)
    print("Num of tokens:", len(vocab))
    return vocab

if __name__ == "__main__":
    main()
