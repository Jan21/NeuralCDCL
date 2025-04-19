import sys
import os
import argparse
import hashlib

import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.raw_data_loader import RawDataLoader
from src.model.registry import CommandRegistry
from src.dataset.dataset import TokenizedDataset
from src.dataset.dataset_builder import DatasetBuilder

parser = argparse.ArgumentParser("Dataset preprocessor. Loads raw data, tokenizes, and saves.")
parser.add_argument("--split", type=str, required=True, help="Dataset split to process (e.g. train, val)")
parser.add_argument("--inspect", action="store_true", help="Print decoded samples")
parser.add_argument("--force", action="store_true", help="Force re-tokenization even if cache exists")
parser.add_argument(
    "--limit", type=int, default=None,
    help="Optional limit on the number of JSON examples to process"
)

cli_args, unknown = parser.parse_known_args()  # `unknown` gets passed to Hydra
sys.argv = [sys.argv[0]] + unknown  # Hydra now sees only unknowns or config overrides


def compute_token_lengths(dataset):
    return [len(sample["input_ids"]) for sample in dataset]


def plot_token_length_histogram(lengths, file_path: str, split: str):
    plt.figure()
    plt.hist(lengths, bins=50, edgecolor="black")
    plt.title(f"Token Length Distribution - {split}")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(file_path)
    print(f"Saved token length histogram: {file_path}")


def inspect_samples(dataset: TokenizedDataset, tokenizer: Tokenizer, split: str, n: int = 3) -> list[str]:
    output_lines = []
    header = f"\n--- Debug info for split: {split} ---"
    print(header)
    output_lines.append(header)

    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]

        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.decode([tid for tid in labels if tid != -100], skip_special_tokens=True)

        block = (
            f"\nExample {i + 1}:\n"
            f"Input IDs:       {input_ids}\n"
            f"Attention Mask:  {attention_mask}\n"
            f"Labels:          {labels}\n"
            f"Decoded Input:   {decoded_input}\n"
            f"Decoded Labels:  {decoded_labels}\n"
        )
        print(block)
        output_lines.append(block)

    return output_lines


def compute_file_checksum(path: Path, algo="sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    split = cli_args.split
    raw_path = Path(to_absolute_path(cfg.data.raw_files[split]))
    out_path = Path(to_absolute_path(cfg.data.tokenized_files[split]))

    if out_path.exists() and not cli_args.force:
        print(f"Found cached tokenized dataset: {out_path}")
        dataset = TokenizedDataset.load(out_path)
    else:
        print(f"Tokenizing split: {split}")
        tokenizer = Tokenizer.from_file(to_absolute_path(cfg.paths.tokenizer))
        registry = CommandRegistry(cfg, tokenizer)
        raw_data_loader = RawDataLoader(cfg)
        raw_data = raw_data_loader.load(split)

        dataset_builder = DatasetBuilder(
            tokenizer, 
            registry, 
            tokenize_batch_size=cfg.data.tokenization.batch_size, 
            num_workers=cfg.data.tokenization.num_workers
        )
        dataset = dataset_builder.build(raw_data)
        dataset.save(out_path)
        print(f"Saved tokenized split '{split}' to: {out_path}")

    if cli_args.inspect:
        info_dir = Path(to_absolute_path(cfg.paths.data)) / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        plot_path = info_dir / f"token_lengths_{split}.png"
        inspect_path = info_dir / f"sample_{split}.txt"

        tokenizer = Tokenizer.from_file(to_absolute_path(cfg.paths.tokenizer))
        inspect_lines = inspect_samples(dataset, tokenizer, split)

        lengths = compute_token_lengths(dataset)
        inspect_lines.append(
            f"[{split.upper()}] Mean: {sum(lengths)/len(lengths):.2f}, Min: {min(lengths)}, Max: {max(lengths)}"
        )
        print(inspect_lines[-1])
        plot_token_length_histogram(lengths, plot_path, split)

        inspect_path.write_text("\n".join(inspect_lines))
        print(f"Saved inspected samples to: {inspect_path}")

        # checksums
        checksums = {
            "raw_file": str(raw_path),
            "raw_checksum": compute_file_checksum(raw_path),
            "tokenized_file": str(out_path),
            "tokenized_checksum": compute_file_checksum(out_path),
        }
        checksum_path = info_dir / f"checksums_{split}.txt"
        checksum_path.write_text("\n".join(f"{k}: {v}" for k, v in checksums.items()))
        print(f"Saved checksums to: {checksum_path}")


if __name__ == "__main__":
    main()
