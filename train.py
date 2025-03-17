# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data import *
import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from callbacks.eval_callback import EvalCallback
from callbacks.save_callback import SaveBeforeEvalCallback
from callbacks.training_callback import TrainingCallback
from config import hf_config
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor

import json
import os
import wandb


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, tokenizer, preprocessor, val_dataset_names, control_tokens, trainer_ckpt_path=None):
        super().__init__()
        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.val_dataset_names = val_dataset_names
        self.trainer_ckpt_path = trainer_ckpt_path
        self.control_tokens = control_tokens
        _, self.hf_conf = hf_config.get_configs(cfg)

    def setup(self, stage):
        self.preprocessor.tokenizer.save_pretrained(self.cfg.convert_hf.in_path)
        with open(os.path.join(self.cfg.convert_hf.in_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def mask_targets(self, input_ids, target_ids):
        """
        Masks target tokens based on predefined start/end tokens.
        """
        module_token = target_ids[:, 1]  # Extracts module-specific token
        mask = torch.ones_like(target_ids, dtype=torch.bool, device=target_ids.device)  # Default: mask all

        def mask_between(start_token, end_token, mask_end_token = True):
            """
            Masks everything between each occurrence of start_token and end_token, separately for each occurrence.
            """
            start_positions = (input_ids == start_token).int()  # 1 at start tokens
            end_positions = (input_ids == end_token).int()  # 1 at end tokens

            # Create segment identifiers for each mask block (each segment gets a unique index)
            segment_ids = torch.cumsum(start_positions, dim=1)

            # Build mask: active only inside valid segments
            active_mask = (segment_ids > 0) & (torch.cumsum(end_positions, dim=1) < segment_ids)
            if mask_end_token:
                active_mask |= end_positions.bool()  # Mask end token positions
            return active_mask

        # Apply masks for solve module
        solve_mask = mask_between(self.control_tokens['solve_tokens']["arguments"], self.control_tokens['solve_tokens']["start"])
        up_mask = mask_between(self.control_tokens['up_tokens']["arguments"], self.control_tokens['up_tokens']["start"])
        up_result_mask = mask_between(self.control_tokens['up_tokens']["results"], self.control_tokens['up_tokens']["end"])
        ac_mask = mask_between(self.control_tokens['ac_tokens']["arguments"], self.control_tokens['ac_tokens']["start"])
        ac_result_mask = mask_between(self.control_tokens['ac_tokens']["results"], self.control_tokens['ac_tokens']["end"])

        # Apply masks only when the module matches
        solve_condition = module_token == self.control_tokens['solve_tokens']["arguments"]
        up_condition = module_token == self.control_tokens['solve_tokens']["arguments"]
        ac_condition = module_token == self.control_tokens['solve_tokens']["arguments"]

        # Update mask based on conditions
        mask = torch.where(solve_condition[:, None], solve_mask | up_result_mask | ac_result_mask, mask)
        mask = torch.where(up_condition[:, None], up_mask, mask)
        mask = torch.where(ac_condition[:, None], ac_mask, mask)

        # Apply the mask to targets, setting masked positions to -100
        return torch.where(mask, torch.tensor(-100, device=target_ids.device), target_ids)

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
                total_steps = (self.cfg.data.num_train / (self.cfg.model.batch_size * self.cfg.model.accumulate_grad_batches)) * self.cfg.model.epochs
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

        for input_ids in inputs:
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
                while True:
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


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    conf, _ = hf_config.get_configs(cfg)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    print("Current model configuration:")
    print(f"n_layer: {cfg.model.n_layer}")
    print(f"n_head: {cfg.model.n_head}")
    print(f"n_embd: {cfg.model.n_embd}")
    print(f"Model name: {cfg.model.name}")

    batch_size = cfg.model.batch_size
    accumulate_grad_batches = cfg.model.accumulate_grad_batches
    num_workers = cfg.data.num_workers
    tokenizer = get_tokenizer(cfg.tok_data)
    control_tokens = {
        'solve_tokens': {
            'arguments': tokenizer.encode("SOLVE_ARGUMENTS", add_special_tokens=False)[0],
            'start': tokenizer.encode("SOLVE_START", add_special_tokens=False)[0],
            'end': tokenizer.encode("SOLVE_END", add_special_tokens=False)[0],
        },
        'up_tokens': {
            'arguments': tokenizer.encode("UNIT_PROPAGATION_ARGUMENTS", add_special_tokens=False)[0],
            'start': tokenizer.encode("UNIT_PROPAGATION_START", add_special_tokens=False)[0],
            'results': tokenizer.encode("UNIT_PROPAGATION_RESULTS", add_special_tokens=False)[0],
            'end': tokenizer.encode("UNIT_PROPAGATION_END", add_special_tokens=False)[0],
        },
        'ac_tokens': {
            'arguments': tokenizer.encode("ANALYZE_CONFLICT_ARGUMENTS", add_special_tokens=False)[0],
            'start': tokenizer.encode("ANALYZE_CONFLICT_START", add_special_tokens=False)[0],
            'results': tokenizer.encode("ANALYZE_CONFLICT_RESULTS", add_special_tokens=False)[0],
            'end': tokenizer.encode("ANALYZE_CONFLICT_END", add_special_tokens=False)[0],
        },
        'eos': tokenizer.encode("[EOS]", add_special_tokens=False)[0],
        'sat': tokenizer.encode("SAT", add_special_tokens=False)[0],
        'unsat': tokenizer.encode("UNSAT", add_special_tokens=False)[0],
    }
    preprocessor = Preprocessor(
        tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    val_dataset_names = ['val', 'test']
    model = LLM(GPT(conf), preprocessor=preprocessor, config=conf)

    lit_model = LitLLM(model=model, tokenizer=tokenizer, cfg=cfg, preprocessor=preprocessor, val_dataset_names=val_dataset_names,
                       control_tokens=control_tokens)
    datasets = get_data(cfg, tokenizer)
    data = Datamodule(datasets, batch_size, num_workers, tokenizer)

    data.connect(max_seq_length=cfg.model.block_size)

    logger = WandbLogger(project=cfg.general.project, name=f"{cfg.general.run_name}", config=wandb_config)

    trainer = L.Trainer(
        accelerator="cuda",
        devices=cfg.general.devices,
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="16-mixed",
        # precision="bf16-true",
        val_check_interval=cfg.eval.val_check_interval,
        callbacks=[TrainingCallback(
            epoch_frequency=cfg.eval.callback_epoch_frequency,
            packed=cfg.data.packed,
            tokenizer=tokenizer,
            control_tokens=control_tokens,
            max_length=cfg.model.block_size,
            acc_sample_size=cfg.eval.callback_acc_data_count,
            val_dataset_names=val_dataset_names)
        ],
        logger=logger,
        log_every_n_steps=cfg.eval.log_step_frequency
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()
