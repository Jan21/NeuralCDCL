########################################################################################################################## EVAL_CALLBACK
from transformers import TrainerCallback
from lightning.pytorch.callbacks import Callback
import os
import json
import torch
import numpy as np
import wandb
import pandas as pd
from datetime import datetime
#from utils.countdown_utils import *
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint


def convert_litgpt_to_hf(cfg):

    out_dir = Path(cfg.convert_hf.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.convert_hf.in_path)
    model_path = out_dir / "pytorch_model.bin"
    model_path = Path(model_path)

    copy_config_files(source_dir=source_dir, out_dir=out_dir)
    convert_lit_checkpoint(checkpoint_dir=source_dir, output_dir=out_dir)

    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        state_dict=state_dict,
        attn_implementation="flash_attention_2",
    )
    return hf_model


class EvalCallback(Callback):
    def __init__(
        self,
        data_dir,
        eval_data,
        tokenizer,
        num_examples=128,
        batch_size=64,
        save_path=None,
        config=None,
        eval_interval=1000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.eval_data = eval_data
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.last_eval_step = -1  # Initialize to -1
        self.config = config
        self.hf_model = None
        self.eval_interval = eval_interval
        self.save_path = save_path

        # Load evaluation data once
        data_file = os.path.join(self.data_dir, self.eval_data)
        with open(data_file, "r") as json_file:
            self.data = json.load(json_file)

        # Create results directory if it doesn't exist
        self.results_dir = self.config.eval.results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize results DataFrame
        self.csv_path = os.path.join(self.results_dir, "eval_results.csv")
        if os.path.exists(self.csv_path):
            self.results_df = pd.read_csv(self.csv_path)
        else:
            self.results_df = pd.DataFrame(
                columns=[
                    "step",
                    "timestamp",
                    "average_rating",
                    "average_true_rating",
                    "accuracy",
                    "true_accuracy",
                    "predictions",
                ]
            )

    def eval_ll(
        self,
        model,
        tokenizer,
        data,
        batch_size=128,
        context_len=4096,
        temperature=0.0,
        n=1,
    ):
        """
        Evaluate the model on the data using a sliding window so that the context length is not exceeded
        """
        output_texts_concat = []
        for b in trange(0, len(data), batch_size):
            batch = data[b : min(b + batch_size, len(data))]
            output_texts = ["" for _ in range(len(batch))]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
            inputs = inputs["input_ids"]

            if n == 1:
                outputs = model.generate(
                    input_ids=inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    max_length=context_len,
                    num_beams=1,
                    do_sample=False,
                )
                output_tokens = outputs
                output_text = tokenizer.batch_decode(
                    output_tokens, skip_special_tokens=False
                )
                tokenizer.padding_side = "left"
                output_texts = [
                    ot + ot_now for ot, ot_now in zip(output_texts, output_text)
                ]
                output_texts_concat += output_texts

        return output_texts_concat

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only run evaluation at specified intervals and if we haven't evaluated at this step
        if (
            trainer.global_step % self.eval_interval == 0
            and trainer.global_step > self.last_eval_step
            and trainer.is_global_zero
        ):
            print(f"Saving model before evaluation...")
            pl_module.llm.model.to(pl_module.llm.preprocessor.device)
            pl_module.llm.save(self.save_path)
            self.run_evaluation(trainer, pl_module)

    def run_evaluation(self, trainer, pl_module):
        print(f"\nRunning custom countdown evaluation at step {trainer.global_step}")

        try:
            self.hf_model = convert_litgpt_to_hf(self.config)
            self.hf_model.cuda()
            self.hf_model.eval()

            # Prepare evaluation data
            test_prompts = [
                self.tokenizer.bos_token
                + f"S {sample['target']} [ {' '.join(map(str,sample['nums']))} ] ,"
                for sample in self.data[: self.num_examples]
            ]
            len_nums = [
                len(sample["nums"]) for sample in self.data[: self.num_examples]
            ]
            data_4 = [d for d, l in zip(test_prompts, len_nums) if l == 4]

            # Get predictions
            predictions = self.eval_ll(
                self.hf_model,
                self.tokenizer,
                data_4,
                batch_size=self.batch_size,
                context_len=4096,
                temperature=0.0,
                n=1,
            )

            # Calculate metrics
            pred_ratings = []
            true_rating = []
            pred_reasons = []

            for i in range(len(predictions)):
                rating, reason = metric_fn(
                    predictions[i].split(self.tokenizer.bos_token)[1], mode="sft"
                )
                tr, _ = metric_fn(f"{self.data[i]['search_path']}", mode="sft")
                pred_ratings.append(rating)
                true_rating.append(tr)
                pred_reasons.append(reason)

            pred_ratings = np.array(pred_ratings)
            avg_rating = float(np.mean(pred_ratings))
            avg_true_rating = float(np.mean(true_rating))
            accuracy = float(np.mean([r > 0 for r in pred_ratings]))
            true_accuracy = float(np.mean([r > 0 for r in true_rating]))

            # Save detailed results
            eval_dir = os.path.join(
                self.config.eval.results_dir, f"step_{trainer.global_step}"
            )
            os.makedirs(eval_dir, exist_ok=True)

            results_file = os.path.join(
                eval_dir,
                f"results_{self.num_examples}_{self.eval_data.replace('/','_')}",
            )
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "trajectories": predictions,
                        "ratings": pred_ratings.tolist(),
                        "reasons": pred_reasons,
                        "test_prompts": test_prompts,
                    },
                    f,
                    indent=4,
                )

            self.last_eval_step = trainer.global_step

            # Log using the trainer's logger instead of wandb directly
            metrics = {
                "countdown_eval/average_rating": avg_rating,
                "countdown_eval/average_true_rating": avg_true_rating,
                "countdown_eval/accuracy": accuracy,
                "countdown_eval/true_accuracy": true_accuracy,
            }

            # Use trainer's logger to log metrics
            for key, value in metrics.items():
                trainer.logger.log_metrics({key: value}, step=trainer.global_step)
            print("Successfully logged countdown evaluation metrics")

            # Save to CSV
            if not any(self.results_df["step"] == trainer.global_step):
                new_row = pd.DataFrame(
                    [
                        {
                            "step": trainer.global_step,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "average_rating": avg_rating,
                            "average_true_rating": avg_true_rating,
                            "accuracy": accuracy,
                            "true_accuracy": true_accuracy,
                            "predictions": json.dumps(predictions),
                        }
                    ]
                )

                self.results_df = pd.concat(
                    [self.results_df, new_row], ignore_index=True
                )
                self.results_df.to_csv(self.csv_path, index=False)

            # Print results summary
            print("\nResults Summary:")
            print(f"Average rating: {avg_rating}")
            print(f"Average true rating: {avg_true_rating}")
            print(f"Accuracy: {accuracy}")
            print(f"True Accuracy: {true_accuracy}")

        except Exception as e:
            print(f"Error during countdown evaluation: {e}")
            raise e

        finally:
            # Cleanup
            del self.hf_model
            torch.cuda.empty_cache()


########################################################################################################################## SAVE CALLBACK
import lightning.pytorch as L
from lightning.pytorch.utilities import rank_zero_only
class SaveBeforeEvalCallback(L.Callback):
    def __init__(self, save_path: str, eval_interval: int):
        self.save_path = save_path
        self.eval_interval = eval_interval
    
    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx):
        """ Save the model before eval_callback is triggered """
        current_step = trainer.global_step
        if current_step % self.eval_interval == 0:
            print(f"Saving model at step {current_step} before evaluation...")
            pl_module.llm.model.to(pl_module.llm.preprocessor.device)
            pl_module.llm.save(self.save_path)




########################################################################################################################## TRAINING CALLBACK
import torch
from lightning.pytorch.callbacks import Callback
import wandb
import random
from lightning.pytorch.loggers import WandbLogger
import numpy as np

class TrainingCallback(Callback):
    def __init__(self, epoch_frequency, packed, tokenizer, control_tokens, max_length, acc_sample_size, val_dataset_names):
        super().__init__()
        self.epoch_frequency = epoch_frequency
        self.packed = packed
        self.tokenizer = tokenizer
        self.control_tokens = control_tokens
        self.max_length = max_length
        self.acc_sample_size = acc_sample_size
        self.val_dataset_names = val_dataset_names

    def extract_segment(self, tokens, start_token, end_token):
        """Extracts relevant token segment."""
        if start_token in tokens and end_token in tokens:
            start_idx = tokens.index(start_token)
            end_idx = tokens.index(end_token) + 1
            return tokens[start_idx:end_idx]
        return []

    def sample_dataset(self, dataset):
        """Randomly samples `acc_sample_size` instances from each module type."""
        solve_samples, up_samples, ac_samples = [], [], []
        dataset = [dataset[i] for i in range(len(dataset))]  # Convert dataset to list
        random.shuffle(dataset)

        for item in dataset:
            input_ids = item["input_ids"]

            if self.control_tokens["solve_tokens"]["arguments"] in input_ids and len(solve_samples) < self.acc_sample_size:
                solve_samples.append(input_ids)
            elif self.control_tokens["up_tokens"]["arguments"] in input_ids and len(up_samples) < self.acc_sample_size:
                up_samples.append(input_ids)
            elif self.control_tokens["ac_tokens"]["arguments"] in input_ids and len(ac_samples) < self.acc_sample_size:
                ac_samples.append(input_ids)

            if len(solve_samples) == self.acc_sample_size and len(up_samples) == self.acc_sample_size and len(ac_samples) == self.acc_sample_size:
                break  # Stop once we have enough samples

        return solve_samples, up_samples, ac_samples

    def process_samples(self, samples, module):
        """Extracts argument and result parts for each sample."""
        args, gt_results = [], []

        for input_ids in samples:
            start_token = self.control_tokens[f"{module}_tokens"]["arguments"]
            start_boundary = self.control_tokens[f"{module}_tokens"]["start"]
            results_token = self.control_tokens[f"{module}_tokens"]["results"] if module != 'solve' else None
            end_token = self.control_tokens[f"{module}_tokens"]["end"]

            args.append(self.extract_segment(input_ids, start_token, start_boundary))
            
            if module == "solve":
                sat_tok = self.control_tokens["sat"]
                unsat_tok = self.control_tokens["unsat"]
                tok = None
                if sat_tok in input_ids and unsat_tok not in input_ids:
                    tok = sat_tok
                elif unsat_tok in input_ids and sat_tok not in input_ids:
                    tok = unsat_tok
                gt_results.append(tok)
            else:
                gt_results.append(self.extract_segment(input_ids, results_token, end_token))  # [RESULTS → END]

        return args, gt_results

    def compute_accuracy(self, pred_lst, trace_gt_lst, module):
        """Computes accuracy based on module type."""
        correct = sum(1 for pred, gt in zip(pred_lst, trace_gt_lst) if pred == gt)
        return correct / len(trace_gt_lst) if trace_gt_lst else 0

    def compute_accuracy_from_predictions(self, solve_preds, up_preds, ac_preds, solve_gt, up_gt, ac_gt):
        """Computes accuracy using existing predictions (no model calls)."""
        return {
            "ANALYZE_CONFLICT": self.compute_accuracy(ac_preds, ac_gt, "ac"),
            "UNIT_PROPAGATION": self.compute_accuracy(up_preds, up_gt, "up"),
            "SOLVE": self.compute_accuracy(solve_preds, solve_gt, "solve"),
        }

    def generate_predictions(self, dataset, model):
        """Generates predictions independently of accuracy."""
        solve_samples, up_samples, ac_samples = self.sample_dataset(dataset)

        # Extract only the argument segments for input
        solve_args, solve_gt = self.process_samples(solve_samples, "solve")
        up_args, up_gt = self.process_samples(up_samples, "up")
        ac_args, ac_gt = self.process_samples(ac_samples, "ac")

        # **Generate Predictions (Single Call)**
        model_generate_fn = model.generate_packed if self.packed else model.generate
        solve_preds_raw = model_generate_fn(solve_args, max_length=self.max_length, stop_token=self.control_tokens['solve_tokens']['end'])
        up_preds_raw = model_generate_fn(up_args, max_length=self.max_length, stop_token=self.control_tokens['up_tokens']['end'])
        ac_preds_raw = model_generate_fn(ac_args, max_length=self.max_length, stop_token=self.control_tokens['ac_tokens']['end'])

        eos_token = self.control_tokens['eos']  # Fetch the EOS token

        # **Convert to lists and append [EOS] token**
        solve_preds = [x[0].tolist() + [eos_token] for x in solve_preds_raw]
        up_preds = [x[0].tolist() + [eos_token] for x in up_preds_raw]
        ac_preds = [x[0].tolist() + [eos_token] for x in ac_preds_raw]

        return solve_preds, up_preds, ac_preds, solve_gt, up_gt, ac_gt, solve_samples, up_samples, ac_samples

    def on_train_epoch_end(self, trainer, model):
        """Logs accuracy and example text at specified frequency."""
        if trainer.current_epoch % self.epoch_frequency == 0:
            val_dataloaders = trainer.val_dataloaders
            for dataloader_idx, val_dataloader in enumerate(val_dataloaders):
                dataset = val_dataloader.dataset
                dataset_name = self.val_dataset_names[dataloader_idx]

                solve_preds, up_preds, ac_preds,solve_gt, up_gt, ac_gt, solve_samples, up_samples, ac_samples = self.generate_predictions(dataset, model)

                # **Extract pred traces for accuracy**
                _, solve_pred_part = self.process_samples(solve_preds, "solve")
                _, up_pred_part = self.process_samples(up_preds, "up")
                _, ac_pred_part = self.process_samples(ac_preds, "ac")

                # **Compute Accuracy**
                module_accuracies = self.compute_accuracy_from_predictions(solve_pred_part, up_pred_part, ac_pred_part, solve_gt, up_gt, ac_gt)

                sample_pred_solve = self.tokenizer.decode(solve_preds[0], skip_special_tokens=True)
                sample_pred_up = self.tokenizer.decode(up_preds[0], skip_special_tokens=True)
                sample_pred_ac = self.tokenizer.decode(ac_preds[0], skip_special_tokens=True)

                sample_gt_solve = self.tokenizer.decode(solve_samples[0], skip_special_tokens=True)
                sample_gt_up = self.tokenizer.decode(up_samples[0], skip_special_tokens=True)
                sample_gt_ac = self.tokenizer.decode(ac_samples[0], skip_special_tokens=True)

                # **Log to WandB**
                if trainer.logger and isinstance(trainer.logger, WandbLogger):
                    wandb_logger = trainer.logger
                    wandb_logger.experiment.log({
                        f"{dataset_name}/accuracy_analyze_conflict": module_accuracies["ANALYZE_CONFLICT"],
                        f"{dataset_name}/accuracy_unit_propagation": module_accuracies["UNIT_PROPAGATION"],
                        f"{dataset_name}/accuracy_solve": module_accuracies["SOLVE"],
                        f"{dataset_name}/sample_solve": wandb.Html(
                            f"<b>Prediction:</b> <p>{sample_pred_solve}</p><br><b>Ground Truth:</b> <p>{sample_gt_solve}</p>"
                        ),
                        f"{dataset_name}/sample_unit_propagation": wandb.Html(
                            f"<b>Prediction:</b> <p>{sample_pred_up}</p><br><b>Ground Truth:</b> <p>{sample_gt_up}</p>"
                        ),
                        f"{dataset_name}/sample_analyze_conflict": wandb.Html(
                            f"<b>Prediction:</b> <p>{sample_pred_ac}</p><br><b>Ground Truth:</b> <p>{sample_gt_ac}</p>"
                        ),
                    })

                print(
                    f"[Epoch {trainer.current_epoch}] Dataset: {dataset_name} "
                    f"AC Acc: {module_accuracies['ANALYZE_CONFLICT']:.3f} | "
                    f"UP Acc: {module_accuracies['UNIT_PROPAGATION']:.3f} | "
                    f"Solve Acc: {module_accuracies['SOLVE']:.3f}\n\n"
                    # f"Sample Pred Solve: {sample_pred_solve}\n"
                    f"Sample Pred UP: {sample_pred_up}\n"
                    f"Sample Pred AC: {sample_pred_ac}\n\n"
                    # f"Sample GT Solve: {sample_gt_solve}\n"
                    f"Sample GT UP: {sample_gt_up}\n"
                    f"Sample GT AC: {sample_pred_ac}\n"
                )