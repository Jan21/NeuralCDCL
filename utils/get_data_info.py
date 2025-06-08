from data_pythia import *
import math
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import glob
@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    tokenizer = get_tokenizer(cfg.tok_data)
    tokenized_datasets = get_data_for_analysis(cfg, tokenizer)
    
    # Analyze token counts for each split
    for split in tokenized_datasets.keys():
        token_counts = [len(sample["input_ids"]) for sample in tokenized_datasets[split]]
        
        # Calculate statistics
        avg_tokens = np.mean(token_counts)
        max_tokens = np.max(token_counts)
        min_tokens = np.min(token_counts)
        
        # Find the sequences with max and min tokens
        max_idx = int(np.argmax(token_counts))  # Convert numpy.int64 to Python int
        min_idx = int(np.argmin(token_counts))  # Convert numpy.int64 to Python int
        
        # Get the original text if needed (before tokenization)
        max_sequence_tokens = tokenized_datasets[split][max_idx]["input_ids"]
        min_sequence_tokens = tokenized_datasets[split][min_idx]["input_ids"]
        
        # Print the statistics
        print(f"\n--- Token Statistics for {split} set ---")
        print(f"Number of samples: {len(token_counts)}")
        print(f"Average token count: {avg_tokens:.2f}")
        print(f"Maximum token count: {max_tokens} (sample index: {max_idx})")
        print(f"Minimum token count: {min_tokens} (sample index: {min_idx})")
        
        # Print token distribution
        print("\nToken count distribution:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"{p}th percentile: {np.percentile(token_counts, p):.1f}")
            
        # Optional: Decode and print the max/min sequences (first 100 tokens)
        print("\nMax token sequence (truncated):")
        print(tokenizer.decode(max_sequence_tokens[:100]) + "...")
        
        print("\nMin token sequence:")
        print(tokenizer.decode(min_sequence_tokens))

if __name__ == "__main__":
    main()