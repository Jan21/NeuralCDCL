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

    def compute_accuracy(self, label_tokens, predicted_tokens):
        correct = sum([
            1 if (self.sat_tok_id in trace_gt and self.sat_tok_id in trace_pr and self.unsat_tok_id not in trace_pr) or
                   (self.unsat_tok_id in trace_gt and self.unsat_tok_id in trace_pr and self.sat_tok_id not in trace_pr)
            else 0
            for trace_gt, trace_pr in zip(label_tokens, predicted_tokens)
        ])
        return correct / len(label_tokens)

    def get_sample_data(self, dataset):
        sample_size = min(self.acc_sample_size, len(dataset))
        sampled_indices = random.sample(range(len(dataset)), sample_size)
        sampled_data = [dataset[i] for i in sampled_indices]
        tokenized_input = [item["input_ids"] for item in sampled_data]

        formula_tokens = [tok_ids[:tok_ids.index(self.formula_end_tok_id) + 1] for tok_ids in tokenized_input]
        formula_max_length = max(len(tokens) for tokens in formula_tokens)
        tokenized_formula_padded = []
        for tokens in formula_tokens:
            padding_length = formula_max_length - len(tokens)
            tokenized_formula_padded.append([self.pad_tok_id] * padding_length + tokens)  # Prepend PAD tokens

        return tokenized_input, tokenized_formula_padded

    def generate_outputs(self, model, padded_formula_tokens):
        inputs = torch.tensor(padded_formula_tokens, device=model.device)
        outputs = model.generate(
            input_ids=inputs,
            max_length=self.max_length
        )

        # Truncate each sequence at the eos_token
        truncated_outputs = []
        for sequence in outputs.cpu().numpy():  # Convert to numpy for easier handling
            # Find the index of the first occurrence of eos_token_id
            eos_index = np.where(sequence == self.trace_end_tok_id)[0]
            if eos_index.size > 0:  # If eos_token_id is found
                truncated_sequence = sequence[:eos_index[0]+1]  # Truncate at the first eos_token
            else:
                truncated_sequence = sequence  # Keep the entire sequence if no eos_token is found
            truncated_outputs.append(truncated_sequence.tolist())  # Convert back to list

        return truncated_outputs

    def on_train_epoch_end(self, trainer, model):
        """Log accuracy and example text for each validation dataset at the specified frequency."""
        if trainer.current_epoch % self.epoch_frequency == 0:
            val_dataloaders = trainer.val_dataloaders
            for dataloader_idx, val_dataloader in enumerate(val_dataloaders):
                dataset = val_dataloader.dataset
                tokenized_input, tokenized_formula_padded = self.get_sample_data(dataset)
                truncated_output_tokens = self.generate_outputs(model, tokenized_formula_padded)

                acc = self.compute_accuracy(tokenized_input, truncated_output_tokens)
                example_text = self.tokenizer.decode(truncated_output_tokens[0], skip_special_tokens=True)

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