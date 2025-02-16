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
from training_callback import TrainingCallback
from config import hf_config
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT, Llama
from litgpt.api import Preprocessor

import json
import os
import wandb


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, val_dataset_names, delimiter_token_id, trainer_ckpt_path=None):
        super().__init__()

        # self.llm = LLM.load(
        #     checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None
        # )
        # return cls(
        #     model=model, preprocessor=preprocessor, prompt_style=prompt_style,
        #     config=config, checkpoint_dir=checkpoint_dir, fabric=fabric, generate_strategy=None,
        #     kv_cache_initialized=False, fixed_kv_cache_size=False
        # )

        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.val_dataset_names = val_dataset_names
        self.trainer_ckpt_path = trainer_ckpt_path
        self.delimiter_token_id = delimiter_token_id
        _, self.hf_conf = hf_config.get_configs(cfg)

    def setup(self, stage):
        self.preprocessor.tokenizer.save_pretrained(self.cfg.convert_hf.in_path)
        with open(os.path.join(self.cfg.convert_hf.in_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def mask_targets(self, input_ids, target_ids):
        # Create a mask for all positions up to and including the first occurrence of search_token_id
        first_search_pos = (input_ids == self.delimiter_token_id).cumsum(dim=1).bool()
        mask = ~first_search_pos.cumsum(dim=1).bool()
        # Apply the mask to targets, setting masked positions to -100
        return torch.where(mask, torch.tensor(-100, device=target_ids.device), target_ids)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets)
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
                total_steps = (self.cfg.data.num_train / self.cfg.eval.batch_size) * self.cfg.model.epochs
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
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Sampling temperature. Lower values make the model more deterministic.

        Returns:
            torch.Tensor: Generated token IDs of shape (batch_size, generated_seq_len).
        """
        self.eval()
        generated = input_ids

        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                logits = self(generated)[:, -1, :]  # (batch_size, vocab_size)

            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated


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
    preprocessor = Preprocessor(
        tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    val_dataset_names=['val_easy', 'val_medium', 'val_hard']
    model = LLM(Llama(conf), preprocessor=preprocessor, config=conf)
    formula_end_token_id = tokenizer.encode("FORMULA_END", add_special_tokens=False)[0]

    lit_model = LitLLM(model=model, cfg=cfg, preprocessor=preprocessor, val_dataset_names=val_dataset_names,
                       delimiter_token_id=formula_end_token_id)
    datasets = get_data(cfg, tokenizer)
    data = Datamodule(datasets, batch_size, num_workers, tokenizer)

    data.connect(max_seq_length=cfg.model.block_size)

    logger = WandbLogger(project=cfg.general.project, name=f"{cfg.general.run_name}", config=wandb_config)

    # eval_callback = EvalCallback(
    #     data_dir=cfg.data.datapath,
    #     eval_data=cfg.data.val_file,
    #     tokenizer=tokenizer,
    #     num_examples=cfg.eval.num_examples,
    #     batch_size=cfg.eval.batch_size,
    #     config=cfg,
    #     eval_interval=cfg.eval.eval_interval,
    #     save_path=cfg.convert_hf.in_path,
    # )

    trainer = L.Trainer(
        accelerator="cuda",
        devices=cfg.general.devices,
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=cfg.eval.val_check_interval,
        callbacks=[TrainingCallback(
            epoch_frequency=cfg.eval.callback_epoch_frequency,
            tokenizer=tokenizer,
            max_length=cfg.model.block_size,
            acc_sample_size=cfg.eval.callback_acc_data_count,
            val_dataset_names=['val_easy', 'val_medium', 'val_hard'])
        ],
        logger=logger,
        log_every_n_steps=cfg.eval.log_step_frequency
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()
