import torch
from lightning.pytorch.callbacks import Callback
import wandb
import random
from lightning.pytorch.loggers import WandbLogger
import numpy as np

class TrainingCallback(Callback):
    def __init__(self, epoch_frequency, tokenizer, max_length, acc_sample_size, val_dataset_names):
        super().__init__()
        self.epoch_frequency = epoch_frequency
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.acc_sample_size = acc_sample_size
        self.val_dataset_names = val_dataset_names

        self.pad_tok_id = tokenizer.convert_tokens_to_ids("[PAD]")
        self.trace_end_tok_id = tokenizer.convert_tokens_to_ids("TRACE_END")
        self.formula_end_tok_id = tokenizer.convert_tokens_to_ids("FORMULA_END")
        self.sat_tok_id = tokenizer.convert_tokens_to_ids("SAT")
        self.unsat_tok_id = tokenizer.convert_tokens_to_ids("UNSAT")

    def compute_accuracy(self, pred_lst, trace_gt_lst):
        correct = sum([
            1 if (self.sat_tok_id in trace_gt and self.sat_tok_id in pred and self.unsat_tok_id not in pred) or
                   (self.unsat_tok_id in trace_gt and self.unsat_tok_id in pred and self.sat_tok_id not in pred)
            else 0
            for pred, trace_gt in zip(pred_lst, trace_gt_lst)
        ])
        return correct / len(trace_gt_lst)

    def get_sample_data(self, dataset):
        sample_size = min(self.acc_sample_size, len(dataset))
        sampled_indices = random.sample(range(len(dataset)), sample_size)
        sampled_data = [dataset[i] for i in sampled_indices]
        tokenized_input = [item["input_ids"] for item in sampled_data]

        formula_tokens_list = []
        trace_tokens_list = []
        for tok_ids in tokenized_input:
            formula_end_idx = tok_ids.index(self.formula_end_tok_id)
            formula_length = formula_end_idx + 1  # Includes FORMULA_END
            formula_tokens = tok_ids[:formula_length]
            trace_tokens = tok_ids[formula_length:]
            formula_tokens_list.append(formula_tokens)
            trace_tokens_list.append(trace_tokens)

        return formula_tokens_list, trace_tokens_list

    def on_train_epoch_end(self, trainer, model):
        """Log accuracy and example text for each validation dataset at the specified frequency."""
        if trainer.current_epoch % self.epoch_frequency == 0:
            val_dataloaders = trainer.val_dataloaders
            for dataloader_idx, val_dataloader in enumerate(val_dataloaders):
                dataset = val_dataloader.dataset
                formula_lst, trace_gt_lst = self.get_sample_data(dataset)
                pred_lst = model.generate(formula_lst, max_length=self.max_length, stop_token=self.trace_end_tok_id)
                acc = self.compute_accuracy(pred_lst, trace_gt_lst)
                example_text = self.tokenizer.decode(pred_lst[0][0], skip_special_tokens=True)

                dataset_name = self.val_dataset_names[dataloader_idx]
                if trainer.logger is not None and isinstance(trainer.logger, WandbLogger):
                    wandb_logger = trainer.logger
                    wandb_logger.experiment.log({
                        f"{dataset_name}/accuracy": acc,
                        f"{dataset_name}/text": wandb.Html(f"<p>{example_text}</p>")
                    })
                print(
                    f"[Epoch {trainer.current_epoch}] "
                    f"Dataset: {dataset_name}; "
                    f"Accuracy: {acc:.3f}; "
                    f"Example Text: \n{example_text}"
                )