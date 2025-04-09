import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from src.data.pipeline import DatasetPipeline
from tokenizers import Tokenizer


def compute_token_lengths(dataset):
    return [len(sample["input_ids"]) for sample in dataset]


def plot_token_length_histogram(lengths, file_path: str, split="train"):
    plt.figure()
    plt.hist(lengths, bins=50, edgecolor="black")
    plt.title(f"Token Length Distribution - {split}")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig(file_path)
    print(f"Saved token length histogram: {file_path}")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    with_subcalls = 'separated_subcalls' if cfg.data.separated_subcalls else ''

    tokenizer = Tokenizer.from_file(to_absolute_path(cfg.paths.tokenizer))
    pipeline = DatasetPipeline(cfg, tokenizer)
    datasets = pipeline.build()

    for split in datasets.keys():
        if split not in datasets:
            print(f"Split '{split}' not found in datasets. Skipping.")
            continue

        # DEBUG: Inspect a few tokenized examples
        print(f"\n--- Debug info for split: {split} ---")
        for i in range(min(3, len(datasets[split]))):
            sample = datasets[split][i]
            input_ids = sample["input_ids"].tolist()
            attention_mask = sample["attention_mask"].tolist()
            labels = sample["labels"].tolist()

            decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_labels = tokenizer.decode([tid for tid in labels if tid != -100], skip_special_tokens=True)

            print(f"\nExample {i + 1}:")
            print("Input IDs:       ", input_ids)
            print("Attention Mask:  ", attention_mask)
            print("Labels:          ", labels)
            print("Decoded Input:   ", decoded_input)
            print("Decoded Labels:  ", decoded_labels)

        lengths = compute_token_lengths(datasets[split])
        print(f"[{split.upper()}] Mean: {sum(lengths) / len(lengths):.2f}, Min: {min(lengths)}, Max: {max(lengths)}")
        file_path = to_absolute_path(os.path.join(cfg.paths.data, f"token_lengths_{split}_{with_subcalls}.png"))
        plot_token_length_histogram(lengths, file_path, split=split)


if __name__ == "__main__":
    main()
