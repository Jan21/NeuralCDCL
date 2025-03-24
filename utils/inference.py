import os
import glob
import json
import torch
import numpy as np
import wandb
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
        'token_full_accuracy': [],
        'exact_match_accuracy': [],
    }
    
    dataset_metrics = {}
    
    for datapath, data in results_dict.items():
        # Initialize metrics for this dataset
        dataset_metrics[datapath] = {
            'token_full_accuracy': [],
            'exact_match_accuracy': [],
        }
        
        gt_solutions_ids = data['gt_solutions_ids']
        predictions_ids = data['predictions_ids']
        
        for i in range(len(gt_solutions_ids)):
            gt_ids = gt_solutions_ids[i]
            pred_ids = predictions_ids[i]

            # Calculate token-by-token accuracy
            min_len = min(len(gt_ids), len(pred_ids))
            matches = 0
            for j in range(min_len):
                if gt_ids[j] == pred_ids[j]:
                    matches += 1
                else:
                    print(f"Not matching: \n GT: \n {gt_ids} \n PRED: \n {pred_ids}")
                    print(f"Decoded: \n GT: \n {tokenizer.decode(gt_ids, skip_special_tokens=True)} \n PRED: \n {tokenizer.decode(pred_ids, skip_special_tokens=True)}")

            token_acc = matches / max(len(gt_ids), len(pred_ids)) if max(len(gt_ids), len(pred_ids)) > 0 else 1.0
            
            # Calculate exact match
            exact_match = 1.0 if gt_ids == pred_ids else 0.0

            
            # Add metrics for this example
            dataset_metrics[datapath]['token_full_accuracy'].append(token_acc)
            dataset_metrics[datapath]['exact_match_accuracy'].append(exact_match)
            
            # Add to overall metrics too
            overall_metrics['token_full_accuracy'].append(token_acc)
            overall_metrics['exact_match_accuracy'].append(exact_match)
    
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

        search_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=False)[0]
        end_token_id = tokenizer.encode("AC-end", add_special_tokens=False)[0]

        # Initialize lists for this dataset
        solutions_text = []
        solutions_ids = []
        prompts_text = []
        prompts_ids = []
        
        for sample in tqdm(test_set):
            input_ids = sample["input_ids"]
            try:
                split_index = input_ids.index(search_token_id)
                end_index = input_ids.index(end_token_id)
            except:
                print(input_ids)
                print(sample)
                print(tokenizer.decode(input_ids, skip_special_tokens=True))
            # Take everything up to "begin" token
            prompt_ids = input_ids[: split_index + 1]

            # Decode to text, add BOS token at start
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            full_prompt = tokenizer.bos_token + " " + prompt_text
            solution_text = tokenizer.decode(input_ids[split_index+1:], skip_special_tokens=True)

            # Re-encode with BOS token
            prompt_with_bos = tokenizer.encode(
                full_prompt, add_special_tokens=False
            )
            solution_ids = input_ids[split_index+1:end_index+1]

            prompts_ids.append(prompt_with_bos)
            prompts_text.append(full_prompt)
            solutions_text.append(solution_text)
            solutions_ids.append(solution_ids)
        
        # Store the lists in the dictionary for this datapath
        results_dict[current_path] = {
            'prompts_ids': prompts_ids,
            'prompts_text': prompts_text,
            'gt_solutions_text': solutions_text,
            'gt_solutions_ids': solutions_ids
        }
        predictions_text = []
        predictions_ids = []

        # Process in batches
        for b in trange(0, len(prompts_ids), batch_size, desc=f"Generating predictions for {os.path.basename(current_path)}"):
            batch = prompts_ids[b : min(b + batch_size, len(prompts_ids))]
            batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
            input_prompt = inputs["input_ids"]
            # print(inputs["attention_mask"])
            # print(inputs["attention_mask"].shape)
            # print(inputs["attention_mask"][0])
            with torch.no_grad():
                outputs = hf_model.generate(
                    input_ids=input_prompt,
                    pad_token_id=tokenizer.pad_token_id,
                    attention_mask=inputs["attention_mask"].to("cuda"),
                    max_length=cfg.model.block_size,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Process each generated sequence
            batch_outputs = outputs.tolist()
            for j, output_ids in enumerate(batch_outputs):
                # Find the search token in the output
                try:
                    split_index = output_ids.index(search_token_id)
                    end_index = output_ids.index(end_token_id)
                except:
                    print(f"Unable to find {end_token_id} or {search_token_id}. Skipping example in {current_path}...")
                    continue
                # Extract everything after the search token
                generated_ids = output_ids[split_index+1:end_index+1]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                predictions_text.append(generated_text)
                predictions_ids.append(generated_ids)

        # Add predictions to the results dictionary
        results_dict[current_path]['predictions_text'] = predictions_text
        results_dict[current_path]['predictions_ids'] = predictions_ids
        # print("GT", solutions_text, "\n\n", "PRED", predictions_text)
        # print("\n\n")
        # print("GT", solutions_ids, "\n\n", "PRED", predictions_ids)

    # After processing all datasets and adding predictions to results_dict:
    output_dir = Path("./temp/generalization_results")
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

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

    # Save the complete results dictionary as pickle
    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"Complete results saved to {output_dir / 'results.pkl'}")    

if __name__ == "__main__":
    main()


