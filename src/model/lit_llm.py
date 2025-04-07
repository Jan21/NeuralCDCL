import torch
import lightning as L
from typing import Optional


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self._cfg = cfg
        self._model = model
        self._tokenizer = tokenizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets_no_mask, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets_no_mask)
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets)
        _, loss = self(idx, targets)
        self.log(f"loss_{self.val_dataset_names[dataloader_idx]}", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return {f"loss_{self.val_dataset_names[dataloader_idx]}": loss, "dataset": self.val_dataset_names[dataloader_idx]}

    def configure_optimizers(self):
        betas = self.cfg.optimizer.betas
        optimizer = torch.optim.AdamW(
            self.llm.model.parameters(), 
            lr=self.cfg.optimizer.lr, 
            weight_decay=self.cfg.optimizer.weight_decay, 
            betas=(betas[0], betas[1])
        )
        # Linear scheduler: warm-up + decay
        def lr_lambda(step):
            if step < self.cfg.optimizer.warmup_steps:
                return (step + 1) / self.cfg.optimizer.warmup_steps  # Warm-up phase
            else:
                # After warm-up, we apply linear decay
                total_steps = (self.num_train / (self.cfg.model.batch_size * self.cfg.model.accumulate_grad_batches)) * self.cfg.model.epochs
                decay_steps = step - self.cfg.optimizer.warmup_steps
                return max(0.0, (total_steps - decay_steps) / total_steps)  # Linear decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.llm(idx, targets)

    def generate(
        self,
        inputs: list[torch.Tensor],
        max_length: int,
        stop_token: int,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """
        Generate text using the model, handling variable input lengths properly.
        """
        self.eval()
        generated_sequences = []

        for input_ids in inputs:
            current_input = torch.tensor(input_ids, device=self.device).unsqueeze(0)  # Shape (1, seq_len)
            generated = current_input

            for _ in range(max_length - current_input.size(1)):
                with torch.no_grad():
                    logits = self(generated)[:, -1, :]  # (1, vocab_size)

                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                generated = torch.cat([generated, next_token], dim=-1)

                if next_token.item() == stop_token:
                    break

            generated_sequences.append(generated)

        return generated_sequences

    def generate_packed(
        self,
        inputs: list[torch.Tensor],
        max_length: int,
        stop_token: int,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """
        Generate text while dynamically adjusting context when encountering UNIT_PROPAGATION_START or ANALYZE_CONFLICT_START.
        """
        self.eval()
        generated_sequences = []

        from tqdm import tqdm
        for input_ids in tqdm(inputs, 'Generating pred'):
            current_input = torch.tensor(input_ids, device=self.device).unsqueeze(0)  # (1, seq_len)
            generated = current_input
            saved_context = None  # To store original context before modifying

            while True:
                for _ in range(max_length - generated.size(1)):
                    with torch.no_grad():
                        logits = self(generated)[:, -1, :]  # (1, vocab_size)

                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                    generated = torch.cat([generated, next_token], dim=-1)

                    if next_token.item() == stop_token:
                        break

                    # Detect transition to UP or AC and store context
                    if next_token.item() in [self.control_tokens["up_tokens"]["start"], self.control_tokens["ac_tokens"]["start"]]:
                        saved_context = generated.clone()  # Save context so far
                        break  # Stop and prepare new context

                if saved_context is None:
                    break  # End generation for this input

                # Prepare new context
                start_token = next_token.item()
                if start_token == self.control_tokens["up_tokens"]["start"]:
                    arguments_token = self.control_tokens["up_tokens"]["arguments"]
                    results_token = self.control_tokens["up_tokens"]["results"]
                    end_token = self.control_tokens["up_tokens"]["end"]
                else:  # ANALYZE_CONFLICT
                    arguments_token = self.control_tokens["ac_tokens"]["arguments"]
                    results_token = self.control_tokens["ac_tokens"]["results"]
                    end_token = self.control_tokens["ac_tokens"]["end"]

                # Extract new context up to and including UNIT_PROPAGATION_ARGUMENTS / ANALYZE_CONFLICT_ARGUMENTS
                arg_indices = (generated == arguments_token).nonzero(as_tuple=True)[1]
                if len(arg_indices) == 0:
                    break  # model error
                arg_index = arg_indices[-1].item()
                new_context = torch.cat([torch.tensor([[self.tokenizer.bos_token_id]], device=self.device), generated[:, arg_index:]], dim=1)

                # Generate continuation with new context
                generated = new_context
                while generated.size(1) < max_length:
                    with torch.no_grad():
                        logits = self(generated)[:, -1, :]
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=-1)

                    if next_token.item() == stop_token or next_token.item() == end_token:
                        break
                
                # Extract results between RESULTS and END
                try:
                    results_start_idx = (generated == results_token).nonzero(as_tuple=True)[1][0].item()
                    results_end_idx = (generated == end_token).nonzero(as_tuple=True)[1][0].item() + 1
                    extracted_results = generated[:, results_start_idx:results_end_idx]
                except IndexError:
                    extracted_results = torch.tensor([], device=self.device).long()  # No valid results found

                # Append results to saved context and continue generating
                generated = torch.cat([saved_context, extracted_results], dim=-1)
                saved_context = None  # Reset context since it has been updated

            generated_sequences.append(generated)

        return generated_sequences