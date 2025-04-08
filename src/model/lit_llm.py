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
        # You probably want one env per input sequence
        outputs = []

        for input_seq in inputs:
            # Initialize environment
            env = AutoregressiveCDCLEnvironment(
                registry=self._command_registry,
                scratchpad=CDCLScratchpad(),  # You may want to pass something pre-initialized
                command_parser=CommandParser(self._command_registry)
            )

            # Init env with the input sequence
            input_ids = input_seq.tolist()
            for tok in input_ids:
                env.append(tok)  # Bootstrap environment

            # Autoregressive generation
            for _ in range(max_length):
                # Get the current sequence
                current_input = env.get_current_input()
                input_tensor = torch.tensor(current_input, dtype=torch.long, device=self.device).unsqueeze(0)

                # Get logits
                logits = self._model(input_tensor)[:, -1, :]  # Shape: [1, vocab_size]
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Append to environment
                env_output = env.append(next_token)

                # Stopping condition
                if next_token == stop_token:
                    break

            final_output = torch.tensor(env.get_current_input(), dtype=torch.long)
            outputs.append(final_output)

        return outputs
