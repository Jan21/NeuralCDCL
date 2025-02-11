import torch
from lightning.pytorch.callbacks import Callback
import wandb
import random

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

    def compute_accuracy(self, model, tokenized_input):
        """Compute accuracy for a given tokenized input."""
        formula_tokens = [tok_ids[:tok_ids.index(self.formula_end_tok_id) + 1] for tok_ids in tokenized_input]
        formula_max_length = max(len(tokens) for tokens in formula_tokens)
        padded_tokens = []
        for tokens in formula_tokens:
            padding_length = formula_max_length - len(tokens)
            padded_tokens.append([self.pad_tok_id] * padding_length + tokens)  # Prepend PAD tokens

        inputs = torch.tensor(padded_tokens, device=model.device)
        outputs = model.generate(
            input_ids=inputs,
            max_length=self.max_length,
            eos_token_id=self.trace_end_tok_id
        )
        correct = sum([
            1 if (self.sat_tok_id in trace_gt and self.sat_tok_id in trace_pr and self.unsat_tok_id not in trace_pr) or
                   (self.unsat_tok_id in trace_gt and self.unsat_tok_id in trace_pr and self.sat_tok_id not in trace_pr)
            else 0
            for trace_gt, trace_pr in zip(tokenized_input, outputs)
        ])
        return correct / len(outputs)

    def generate_example_text(self, model, tokenized_input):
        """Generate example text for a given tokenized input."""
        formula_tokens = [tok_ids[:tok_ids.index(self.formula_end_tok_id) + 1] for tok_ids in tokenized_input]
        formula_max_length = max(len(tokens) for tokens in formula_tokens)
        padded_tokens = []
        for tokens in formula_tokens:
            padding_length = formula_max_length - len(tokens)
            padded_tokens.append([self.pad_tok_id] * padding_length + tokens)  # Prepend PAD tokens

        example_input = torch.tensor(padded_tokens[0], device=model.device)
        output = model.generate(
            input_ids=example_input.unsqueeze(0),
            max_length=self.max_length,
            eos_token_id=self.trace_end_tok_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def on_train_epoch_end(self, trainer, model):
        """Log accuracy and example text for each validation dataset at the specified frequency."""
        if trainer.current_epoch % self.epoch_frequency == 0 and trainer.current_epoch > 0:
            val_dataloaders = trainer.val_dataloaders
            for dataloader_idx, val_dataloader in enumerate(val_dataloaders):
                dataset = val_dataloader.dataset

                # Sample 100 random entries from the dataset
                sample_size = min(self.acc_sample_size, len(dataset))
                sampled_indices = random.sample(range(len(dataset)), sample_size)
                sampled_data = [dataset[i] for i in sampled_indices]

                tokenized_input = [item["input_ids"] for item in sampled_data]

                acc = self.compute_accuracy(model, tokenized_input)
                example_text = self.generate_example_text(model, tokenized_input)

                dataset_name = self.val_dataset_names[dataloader_idx]
                wandb.log({
                    f"{dataset_name}/accuracy": acc,
                    f"{dataset_name}/text": wandb.Html(f"<p>{example_text}</p>")
                })
                print(
                    f"[Epoch {trainer.current_epoch}] "
                    f"Dataset: {dataset_name}; "
                    f"Accuracy: {acc:.3f}; "
                    f"Example Text: \n{example_text}"
                )