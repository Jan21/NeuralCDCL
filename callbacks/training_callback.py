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
                print(self.tokenizer.decode(input_ids, skip_special_tokens=True)[-30:])
                print()
            elif self.control_tokens["up_tokens"]["arguments"] in input_ids and len(up_samples) < self.acc_sample_size:
                up_samples.append(input_ids)
            elif self.control_tokens["ac_tokens"]["arguments"] in input_ids and len(ac_samples) < self.acc_sample_size:
                ac_samples.append(input_ids)

            if len(solve_samples) == self.acc_sample_size and len(up_samples) == self.acc_sample_size and len(ac_samples) == self.acc_sample_size:
                break  # Stop once we have enough samples
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx')

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
        if module == 'solve':
            print(pred_lst, trace_gt_lst)
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

        return solve_preds, up_preds, ac_preds, solve_gt, up_gt, ac_gt

    def on_train_epoch_end(self, trainer, model):
        """Logs accuracy and example text at specified frequency."""
        if trainer.current_epoch % self.epoch_frequency == 0:
            val_dataloaders = trainer.val_dataloaders
            for dataloader_idx, val_dataloader in enumerate(val_dataloaders):
                dataset = val_dataloader.dataset
                dataset_name = self.val_dataset_names[dataloader_idx]
                print(dataset_name)

                solve_preds, up_preds, ac_preds,solve_gt, up_gt, ac_gt = self.generate_predictions(dataset, model)

                # **Extract pred traces for accuracy**
                _, solve_pred_part = self.process_samples(solve_preds, "solve")
                _, up_pred_part = self.process_samples(up_preds, "up")
                _, ac_pred_part = self.process_samples(ac_preds, "ac")

                # **Compute Accuracy**
                module_accuracies = self.compute_accuracy_from_predictions(solve_pred_part, up_pred_part, ac_pred_part, solve_gt, up_gt, ac_gt)

                # **Select and Decode Random Samples**
                random_solve = self.tokenizer.decode(random.choice(solve_preds), skip_special_tokens=True)
                random_up = self.tokenizer.decode(random.choice(up_preds), skip_special_tokens=True) if len(up_preds) else ""
                random_ac = self.tokenizer.decode(random.choice(ac_preds), skip_special_tokens=True) if len(ac_preds) else ""

                # **Log to WandB**
                if trainer.logger and isinstance(trainer.logger, WandbLogger):
                    wandb_logger = trainer.logger
                    wandb_logger.experiment.log({
                        f"{dataset_name}/accuracy_analyze_conflict": module_accuracies["ANALYZE_CONFLICT"],
                        f"{dataset_name}/accuracy_unit_propagation": module_accuracies["UNIT_PROPAGATION"],
                        f"{dataset_name}/accuracy_solve": module_accuracies["SOLVE"],
                        f"{dataset_name}/sample_solve": wandb.Html(f"<p>{random_solve}</p>"),
                        f"{dataset_name}/sample_up": wandb.Html(f"<p>{random_up}</p>"),
                        f"{dataset_name}/sample_ac": wandb.Html(f"<p>{random_ac}</p>"),
                    })

                print(
                    f"[Epoch {trainer.current_epoch}] Dataset: {dataset_name} "
                    f"AC Acc: {module_accuracies['ANALYZE_CONFLICT']:.3f} | "
                    f"UP Acc: {module_accuracies['UNIT_PROPAGATION']:.3f} | "
                    f"Solve Acc: {module_accuracies['SOLVE']:.3f}\n"
                    f"Sample Solve: {random_solve}\n"
                    f"Sample UP: {random_up}\n"
                    f"Sample AC: {random_ac}\n"
                )