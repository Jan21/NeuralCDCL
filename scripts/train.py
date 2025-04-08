import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
import hydra
from hydra.utils import to_absolute_path
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor

from tokenizers import Tokenizer
from src.dataset.dataset_pipeline import DatasetPipeline
from src.dataset.datamodule import Datamodule
from src.model.lit_llm import LitLLM
from src.model.command_registry import CommandRegistry

import wandb


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Tokenizer.
    tokenizer = Tokenizer.from_file(to_absolute_path(cfg.paths.tokenizer))

    # Data.
    pipeline = DatasetPipeline(cfg, tokenizer)
    datasets = pipeline.build()
    dataloaders = pipeline.build_dataloaders(datasets)
    datamodule = Datamodule(dataloaders=dataloaders)
    
    CommandRegistry(cfg, tokenizer)

    exit()

    # LitGPT.
    lit_cfg = Config(
        name="cdcl-pythia",
        block_size=cfg.train.model.block_size,
        n_layer=cfg.train.model.n_layer,
        n_head=cfg.train.model.n_head,
        n_embd=cfg.train.model.n_embd,
        padded_vocab_size=tokenizer.vocab_size,
        intermediate_size=cfg.train.model.n_embd * 4,
        padding_multiple=128,  # pads vocab to multiples of 128
    )
    preprocessor = Preprocessor(tokenizer, device="cpu")
    model = LLM(GPT(lit_cfg), preprocessor=preprocessor, config=lit_cfg)

    # Custom LLM adapter.
    lit_model = LitLLM(model=model, tokenizer=tokenizer, cfg=cfg, preprocessor=preprocessor, val_dataset_names=val_dataset_names,
                       control_tokens=control_tokens, num_train=num_train)


    flattened_cfg = OmegaConf.to_container(cfg, resolve=True)
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
