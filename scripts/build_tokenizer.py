import sys
import os
import argparse
import re

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.raw_data_loader import RawDataLoader

parser = argparse.ArgumentParser(description="Tokenizer builder.")
parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to for infering the vocabulary")
parser.add_argument("--split", type=str, default='ood', help="Which data file to use.")
cli_args, unknown = parser.parse_known_args()  # `unknown` gets passed to Hydra
sys.argv = [sys.argv[0]] + unknown  # Hydra now sees only unknowns or config overrides


def save_tokenizer(vocab: list[str], save_path: str, special_tokens=None, save_vocab_txt=True):
    if special_tokens is None:
        special_tokens = ['[PAD]']

    vocab_dict = {s: i for i, s in enumerate(set(vocab + special_tokens))}
    tokenizer = Tokenizer(WordLevel(vocab_dict))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(special_tokens)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)

    # Save vocab.txt (optional)
    if save_vocab_txt:
        vocab_txt_path = os.path.join(os.path.dirname(save_path), "vocab.txt")
        with open(vocab_txt_path, "w") as f:
            for token in sorted(vocab_dict, key=vocab_dict.get):  # sorted by token ID
                f.write(token + "\n")

    return tokenizer

def build_vocab_from_texts(texts: List[str]) -> List[str]:
    pattern = r'\[|\]|-?x(?:1[0-9]|2[0-5]|[1-9])\b|[^\[\]\s]+'
    all_text = " ".join(texts)
    tokens = re.findall(pattern, all_text)
    vocab = sorted(set(tokens))
    return vocab

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    raw_data = RawDataLoader(cfg).load(cli_args.split)

    if cli_args.num_examples is not None:
        raw_data = raw_data[:cli_args.num_examples]

    texts = [
        item if isinstance(item, str) else log
        for data_point in raw_data
        for item in data_point.values()
        for log in (item if not isinstance(item, str) else [item])
    ]

    vocab = build_vocab_from_texts(texts)
    print(f"Vocab size: {len(vocab)}")

    save_tokenizer(vocab, to_absolute_path(cfg.paths.tokenizer))

if __name__ == "__main__":
    main()