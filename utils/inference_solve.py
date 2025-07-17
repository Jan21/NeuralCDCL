import os
import glob
import json
import torch
import numpy as np
import wandb
import pickle
import random
from datetime import datetime
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from data_pythia import *

def calculate_metrics(results_dict, tokenizer):
    overall_metrics = {
        'token_accuracy_1': [],
        'exact_match_1': [],
        'token_accuracy_2': [],
        'exact_match_2': [],
        'full_exact_match': [],
    }
    
    dataset_metrics = {}
    
    for datapath, data in results_dict.items():
        # Initialize metrics for this dataset
        dataset_metrics[datapath] = {
            'token_accuracy_1': [],
            'exact_match_1': [],
            'token_accuracy_2': [],
            'exact_match_2': [],
            'full_exact_match': [],
        }
        
        gt_part1_ids = data['gt_part1_ids']
        gt_part2_ids = data['gt_part2_ids']
        pred_part1_ids = data['pred_part1_ids']
        pred_part2_ids = data['pred_part2_ids']
        
        # Ensure all lists have same length
        min_len = min(len(gt_part1_ids), len(gt_part2_ids), len(pred_part1_ids), len(pred_part2_ids))
        gt_part1_ids = gt_part1_ids[:min_len]
        gt_part2_ids = gt_part2_ids[:min_len]
        pred_part1_ids = pred_part1_ids[:min_len]
        pred_part2_ids = pred_part2_ids[:min_len]
        
        print(f"Processing {min_len} samples for {os.path.basename(datapath)}")
        
        valid_for_metrics = 0
        samples_shown = 0
        
        for i in range(min_len):
            # Skip samples where either ground truth or predictions are empty (placeholders)
            # For single-part examples, Part 1 will be empty, so we only check Part 2
            gt_ids_1 = gt_part1_ids[i] 
            pred_ids_1 = pred_part1_ids[i]
            gt_ids_2 = gt_part2_ids[i]
            pred_ids_2 = pred_part2_ids[i]
            
            # Skip if Part 2 is empty (main content), but Part 1 can be empty for single-part examples
            if (len(gt_ids_2) == 0 or len(pred_ids_2) == 0):
                continue
            
            valid_for_metrics += 1
            
            # Part 1 metrics (handle empty Part 1 for single-part examples)
            if len(gt_ids_1) == 0 and len(pred_ids_1) == 0:
                # Both empty (single-part example) - perfect match
                token_acc_1 = 1.0
                exact_match_1 = 1.0
            elif len(gt_ids_1) == 0 or len(pred_ids_1) == 0:
                # One empty, one not - no match
                token_acc_1 = 0.0
                exact_match_1 = 0.0
            else:
                # Both non-empty - calculate normally
                min_len_1 = min(len(gt_ids_1), len(pred_ids_1))
                matches_1 = sum(1 for j in range(min_len_1) if gt_ids_1[j] == pred_ids_1[j])
                token_acc_1 = matches_1 / max(len(gt_ids_1), len(pred_ids_1))
                exact_match_1 = 1.0 if gt_ids_1 == pred_ids_1 else 0.0
            
            # Part 2 metrics (always non-empty)
            min_len_2 = min(len(gt_ids_2), len(pred_ids_2))
            matches_2 = sum(1 for j in range(min_len_2) if gt_ids_2[j] == pred_ids_2[j])
            token_acc_2 = matches_2 / max(len(gt_ids_2), len(pred_ids_2))
            exact_match_2 = 1.0 if gt_ids_2 == pred_ids_2 else 0.0
            
            # Full exact match using token ID comparison (concatenate both parts)
            gt_full_ids = gt_ids_1 + gt_ids_2
            pred_full_ids = pred_ids_1 + pred_ids_2
            full_exact_match = 1.0 if gt_full_ids == pred_full_ids else 0.0
            
            # Show exactly 5 samples
            if samples_shown < 5:
                sample_type = "Two-part" if len(gt_ids_1) > 0 else "Single-part"
                print(f"\n--- Sample {samples_shown + 1} ({sample_type}) ---")
                if len(gt_ids_1) > 0:
                    print(f"GT Part 1: {tokenizer.decode(gt_ids_1, skip_special_tokens=False)}")
                    print(f"Pred Part 1: {tokenizer.decode(pred_ids_1, skip_special_tokens=False)}")
                else:
                    print("Part 1: Empty (single-part example)")
                print(f"GT Part 2: {tokenizer.decode(gt_ids_2, skip_special_tokens=False)}")
                print(f"Pred Part 2: {tokenizer.decode(pred_ids_2, skip_special_tokens=False)}")
                samples_shown += 1
            
            # Add metrics for this example
            dataset_metrics[datapath]['token_accuracy_1'].append(token_acc_1)
            dataset_metrics[datapath]['exact_match_1'].append(exact_match_1)
            dataset_metrics[datapath]['token_accuracy_2'].append(token_acc_2)
            dataset_metrics[datapath]['exact_match_2'].append(exact_match_2)
            dataset_metrics[datapath]['full_exact_match'].append(full_exact_match)
            
            # Add to overall metrics too
            overall_metrics['token_accuracy_1'].append(token_acc_1)
            overall_metrics['exact_match_1'].append(exact_match_1)
            overall_metrics['token_accuracy_2'].append(token_acc_2)
            overall_metrics['exact_match_2'].append(exact_match_2)
            overall_metrics['full_exact_match'].append(full_exact_match)
        
        print(f"Used {valid_for_metrics} valid samples for metrics calculation")
    
    # Calculate averages for each dataset
    for datapath in dataset_metrics:
        for metric in dataset_metrics[datapath]:
            dataset_metrics[datapath][metric] = np.mean(dataset_metrics[datapath][metric]) if dataset_metrics[datapath][metric] else 0
    
    # Calculate overall averages
    for metric in overall_metrics:
        overall_metrics[metric] = np.mean(overall_metrics[metric]) if overall_metrics[metric] else 0
    
    return overall_metrics, dataset_metrics

@hydra.main(
    config_path="../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    
    batch_size = cfg.inference.batch_size
    num_workers = cfg.data.num_workers
    # get hf model for batch inference
    model_dir = Path(f"{cfg.inference.modelpath}")
    state_dict = torch.load(model_dir / "model.pth")
    hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            state_dict=state_dict,
            attn_implementation="flash_attention_2",
        )

    hf_model.cuda()
    hf_model.eval()
    # get hf model's tokenizer
    tokenizer = get_tokenizer(cfg.tok_data)

    # Define all required token IDs
    search_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=False)[0]
    end_token_id = tokenizer.encode("END", add_special_tokens=False)[0]
    read_learned_clause_token_id = tokenizer.encode("READ_LEARNED_CLAUSE", add_special_tokens=False)[0]
    read_begin_token_id = tokenizer.encode("READ_BEGIN", add_special_tokens=False)[0]
    delimiter_token_id = tokenizer.encode("SPLIT_BEGIN", add_special_tokens=False)[0]
    read_end_token_id = tokenizer.encode("READ_END", add_special_tokens=False)[0]
    semicolon_token_id = tokenizer.encode(";", add_special_tokens=False)[0]

    # load the data from a directory
    datapaths = glob.glob(f"{cfg.inference.datapath}/*.json")
    # tokenize it
    tokenized_datasets = get_data_for_inference(cfg, datapaths, tokenizer)

    results_dict = {}
    # for all tokenized datasets
    for i, tok_dataset in tqdm(enumerate(tokenized_datasets)):
        # Get the corresponding datapath
        current_path = datapaths[i]
        
        data = Datamodule(tok_dataset, batch_size, num_workers, tokenizer)
        data.connect(max_seq_length=cfg.model.block_size)
        data.setup()

        test_set = data.test_dataset

        # Initialize lists for this dataset
        prompts_part1_ids = []
        prompts_part1_text = []
        prompts_part2_ids = []
        prompts_part2_text = []
        gt_part1_ids = []
        gt_part1_text = []
        gt_part2_ids = []
        gt_part2_text = []
        
        valid_samples = 0
        total_samples = 0
        two_part_count = 0
        single_part_count = 0
        
        for sample in tqdm(test_set):
            input_ids = sample["input_ids"]
            total_samples += 1
            valid_sample = True
            is_two_part = False
            read_learned_clause_index = None
            split_index = None
            end_index = None
            semicolon_index = None
            
            try:
                # Find required token positions
                split_index = input_ids.index(search_token_id)
                end_index = input_ids.index(end_token_id)
                
                # Check if this is a two-part example (has READ_LEARNED_CLAUSE) or single-part
                try:
                    read_learned_clause_index = input_ids.index(read_learned_clause_token_id)
                    is_two_part = True
                except ValueError:
                    is_two_part = False
                
                if is_two_part:
                    # Two-part example: Part 1 (READ_LEARNED_CLAUSE) + Part 2 (semicolon to END)
                    # Find READ_END token that comes after READ_LEARNED_CLAUSE
                    read_end_index = None
                    for j in range(read_learned_clause_index + 1, len(input_ids)):
                        if input_ids[j] == read_end_token_id:
                            read_end_index = j
                            break
                    
                    if read_end_index is None:
                        valid_sample = False
                    else:
                        # Find first semicolon after READ_END token
                        semicolon_index = None
                        for j in range(read_end_index + 1, len(input_ids)):
                            if input_ids[j] == semicolon_token_id:
                                semicolon_index = j
                                break
                        
                        if semicolon_index is None:
                            valid_sample = False
                else:
                    # Single-part example: No Part 1, Part 2 from first semicolon after SPLIT_BEGIN to END
                    # Find first semicolon after SPLIT_BEGIN
                    semicolon_index = None
                    for j in range(split_index + 1, len(input_ids)):
                        if input_ids[j] == semicolon_token_id:
                            semicolon_index = j
                            break
                    
                    if semicolon_index is None:
                        valid_sample = False
                        
            except ValueError:
                valid_sample = False
            
            if valid_sample:
                valid_samples += 1
                
                if is_two_part:
                    two_part_count += 1
                    # Two-part example
                    # Part 1: from search_token_id+1 to read_learned_clause_token_id (including READ_LEARNED_CLAUSE)
                    prompt_part1_ids = input_ids[:split_index + 1]
                    gt_part1_ids_sample = input_ids[split_index + 1:read_learned_clause_index + 1]
                    
                    # Part 2: from semicolon to end_token_id (including both semicolon and END)
                    prompt_part2_ids = input_ids[:semicolon_index]  # Exclude semicolon from prompt
                    gt_part2_ids_sample = input_ids[semicolon_index:end_index + 1]  # GT starts with semicolon
                else:
                    single_part_count += 1
                    # Single-part example (no READ_LEARNED_CLAUSE)
                    # Part 1: empty
                    prompt_part1_ids = []
                    gt_part1_ids_sample = []
                    
                    # Part 2: from semicolon to end_token_id (including both semicolon and END)
                    prompt_part2_ids = input_ids[:semicolon_index]  # Exclude semicolon from prompt
                    gt_part2_ids_sample = input_ids[semicolon_index:end_index + 1]  # GT starts with semicolon
                
                # Convert to text with BOS token for prompts
                if len(prompt_part1_ids) > 0:
                    prompt_part1_text_str = tokenizer.bos_token + " " + tokenizer.decode(prompt_part1_ids, skip_special_tokens=True)
                    prompt_part1_with_bos = tokenizer.encode(prompt_part1_text_str, add_special_tokens=False)
                else:
                    prompt_part1_text_str = ""
                    prompt_part1_with_bos = []
                
                prompt_part2_text_str = tokenizer.bos_token + " " + tokenizer.decode(prompt_part2_ids, skip_special_tokens=True)
                gt_part1_text_str = tokenizer.decode(gt_part1_ids_sample, skip_special_tokens=False) if len(gt_part1_ids_sample) > 0 else ""
                gt_part2_text_str = tokenizer.decode(gt_part2_ids_sample, skip_special_tokens=False)
                
                # Re-encode prompts with BOS token
                prompt_part2_with_bos = tokenizer.encode(prompt_part2_text_str, add_special_tokens=False)
                
                prompts_part1_ids.append(prompt_part1_with_bos)
                prompts_part1_text.append(prompt_part1_text_str)
                prompts_part2_ids.append(prompt_part2_with_bos)
                prompts_part2_text.append(prompt_part2_text_str)
                gt_part1_ids.append(gt_part1_ids_sample)
                gt_part1_text.append(gt_part1_text_str)
                gt_part2_ids.append(gt_part2_ids_sample)
                gt_part2_text.append(gt_part2_text_str)
            else:
                # Add placeholders to maintain alignment
                prompts_part1_ids.append([])
                prompts_part1_text.append("")
                prompts_part2_ids.append([])
                prompts_part2_text.append("")
                gt_part1_ids.append([])
                gt_part1_text.append("")
                gt_part2_ids.append([])
                gt_part2_text.append("")
        
        print(f"Dataset {os.path.basename(current_path)}: {valid_samples}/{total_samples} valid samples ({100*valid_samples/total_samples:.1f}%)")
        print(f"  Two-part examples: {two_part_count}, Single-part examples: {single_part_count}")
        
        # Store the ground truth data
        results_dict[current_path] = {
            'prompts_part1_ids': prompts_part1_ids,
            'prompts_part1_text': prompts_part1_text,
            'prompts_part2_ids': prompts_part2_ids,
            'prompts_part2_text': prompts_part2_text,
            'gt_part1_ids': gt_part1_ids,
            'gt_part1_text': gt_part1_text,
            'gt_part2_ids': gt_part2_ids,
            'gt_part2_text': gt_part2_text
        }
        
        pred_part1_ids = []
        pred_part1_text = []
        pred_part2_ids = []
        pred_part2_text = []

        # First inference: Part 1 (generate up to READ_LEARNED_CLAUSE token)
        print(f"Running Part 1 inference for {os.path.basename(current_path)}")
        for b in trange(0, len(prompts_part1_ids), batch_size, desc="Part 1 Generation"):
            batch = prompts_part1_ids[b : min(b + batch_size, len(prompts_part1_ids))]
            
            # Filter out empty prompts
            valid_batch = []
            batch_indices = []
            for idx, prompt in enumerate(batch):
                if len(prompt) > 0:
                    valid_batch.append(prompt)
                    batch_indices.append(idx)
            
            # Initialize predictions for this batch
            batch_pred_ids = [[] for _ in range(len(batch))]
            batch_pred_text = ["" for _ in range(len(batch))]
            
            if len(valid_batch) > 0:
                batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in valid_batch]
                tokenizer.padding_side = "left"
                inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
                
                with torch.no_grad():
                    outputs = hf_model.generate(
                        input_ids=inputs["input_ids"],
                        pad_token_id=tokenizer.pad_token_id,
                        attention_mask=inputs["attention_mask"],
                        max_length=4096,
                        num_beams=1,
                        do_sample=False,
                        eos_token_id=read_learned_clause_token_id,  # Stop at READ_LEARNED_CLAUSE
                    )
                
                # Process valid outputs
                for valid_idx, output_ids in enumerate(outputs.tolist()):
                    batch_idx = batch_indices[valid_idx]
                    try:
                        split_index = output_ids.index(search_token_id)
                        # Extract everything after search token up to and including READ_LEARNED_CLAUSE
                        try:
                            end_index = output_ids.index(read_learned_clause_token_id)
                            generated_ids = output_ids[split_index + 1:end_index + 1]
                        except ValueError:
                            # If READ_LEARNED_CLAUSE not found, take everything after search token
                            generated_ids = output_ids[split_index + 1:]
                        
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                        batch_pred_ids[batch_idx] = generated_ids
                        batch_pred_text[batch_idx] = generated_text
                    except ValueError:
                        # Keep empty placeholder
                        pass
            
            # Add batch results to main lists
            pred_part1_ids.extend(batch_pred_ids)
            pred_part1_text.extend(batch_pred_text)

        # Second inference: Part 2 (generate from semicolon to END)
        print(f"Running Part 2 inference for {os.path.basename(current_path)}")
        for b in trange(0, len(prompts_part2_ids), batch_size, desc="Part 2 Generation"):
            batch = prompts_part2_ids[b : min(b + batch_size, len(prompts_part2_ids))]
            
            # Filter out empty prompts
            valid_batch = []
            batch_indices = []
            for idx, prompt in enumerate(batch):
                if len(prompt) > 0:
                    valid_batch.append(prompt)
                    batch_indices.append(idx)
            
            # Initialize predictions for this batch
            batch_pred_ids = [[] for _ in range(len(batch))]
            batch_pred_text = ["" for _ in range(len(batch))]
            
            if len(valid_batch) > 0:
                batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in valid_batch]
                tokenizer.padding_side = "left"
                inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
                
                with torch.no_grad():
                    outputs = hf_model.generate(
                        input_ids=inputs["input_ids"],
                        pad_token_id=tokenizer.pad_token_id,
                        attention_mask=inputs["attention_mask"],
                        max_length=4096,
                        num_beams=1,
                        do_sample=False,
                        eos_token_id=end_token_id,  # Stop at END
                    )
                
                # Process valid outputs
                for valid_idx, output_ids in enumerate(outputs.tolist()):
                    batch_idx = batch_indices[valid_idx]
                    try:
                        # Extract everything the model generated after the prompt
                        prompt_len = len(inputs["input_ids"][valid_idx])
                        
                        try:
                            end_index = output_ids.index(end_token_id)
                            generated_ids = output_ids[prompt_len:end_index + 1]
                        except ValueError:
                            # If END not found, take everything after prompt
                            generated_ids = output_ids[prompt_len:]
                        
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                        batch_pred_ids[batch_idx] = generated_ids
                        batch_pred_text[batch_idx] = generated_text
                    except:
                        # Keep empty placeholder
                        pass
            
            # Add batch results to main lists
            pred_part2_ids.extend(batch_pred_ids)
            pred_part2_text.extend(batch_pred_text)

        # Add predictions to results dictionary
        results_dict[current_path]['pred_part1_ids'] = pred_part1_ids
        results_dict[current_path]['pred_part1_text'] = pred_part1_text
        results_dict[current_path]['pred_part2_ids'] = pred_part2_ids
        results_dict[current_path]['pred_part2_text'] = pred_part2_text

    # Calculate metrics
    overall_metrics, dataset_metrics = calculate_metrics(results_dict, tokenizer)

    # Add metrics to results_dict
    results_dict['overall_metrics'] = overall_metrics
    results_dict['dataset_metrics'] = dataset_metrics

    # Print metrics
    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nDataset Metrics:")
    for datapath, metrics in dataset_metrics.items():
        print(f"\n{os.path.basename(datapath)}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save results
    output_dir = Path("./temp/generalization_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results_two_part.pkl", 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"Complete results saved to {output_dir / 'results_two_part.pkl'}")
    
    # Logging
    model_name = model_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = f"{model_name}_two_part_metrics.log"
    
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] Two-part Evaluation for {model_name}\n")
        f.write("Overall Metrics:\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("Dataset Metrics:\n")
        for datapath, metrics in dataset_metrics.items():
            f.write(f"{os.path.basename(datapath)}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        f.write("-" * 50 + "\n")

    print(f"Metrics logged to {log_file}")

if __name__ == "__main__":
    main()