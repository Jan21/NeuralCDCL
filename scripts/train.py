import sys
import os

import torch
from litgpt import LLM
import lightning as L
import hydra
from hydra.utils import to_absolute_path
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.api import Preprocessor
from lightning.pytorch.callbacks import ModelCheckpoint

import random
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataset import TokenizedDataset
from src.dataset.dataloader_builder import DataloaderBuilder
from src.model.registry import CommandRegistry
from src.model.callbacks.eval import EvalCallback
from src.model.callbacks.inference import InferenceCallback
from src.model.lit_wrapper import LitWrapper


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.set_num_threads(os.cpu_count())

    # For reproducibility.
    random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)

    # Set device.
    if cfg.general.accelerator == "cpu":
        device = torch.device("cpu")
    elif cfg.general.accelerator in ("gpu", "cuda", "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif cfg.general.accelerator == "mps":
        device = torch.device("mps")
    else:
        raise ValueError(f"Unsupported accelerator: {cfg.general.accelerator}")

    # Save config.
    config_path = to_absolute_path(cfg.paths.config_export)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    OmegaConf.save(cfg, config_path)

    # Tokenizer.
    tokenizer = Tokenizer.from_file(to_absolute_path(cfg.paths.tokenizer))
    registry = CommandRegistry(cfg, tokenizer)

    # Data.
    datasets = {
        split: TokenizedDataset.load(path).filter_by_block_size(max_len=cfg.train.model.block_size)
        for split, path in cfg.data.tokenized_files.items()
    }
    loader_builder = DataloaderBuilder(
        batch_size=cfg.data.dataloader.batch_size,
        registry=registry,
        num_workers=cfg.data.dataloader.num_workers,
        device=device
    )
    dataloaders = {
        split: loader_builder.build_dataloader(dataset, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    
    # LLM config.
    lit_cfg = Config(
        name=cfg.general.run_name,
        block_size=cfg.train.model.block_size,
        n_layer=cfg.train.model.n_layer,
        n_head=cfg.train.model.n_head,
        n_embd=cfg.train.model.n_embd,
        padded_vocab_size=tokenizer.get_vocab_size(),
        intermediate_size=cfg.train.model.n_embd * 4,
        padding_multiple=128,  # pads vocab to multiples of 128
    )
    preprocessor = Preprocessor(tokenizer, device="cpu")
    llm = LLM(GPT(lit_cfg), preprocessor=preprocessor, config=lit_cfg)
    model = LitWrapper(llm, cfg)

    # Wandb config.
    flattened_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(project=cfg.general.project, name=f"{cfg.general.run_name}", config=flattened_cfg)

    # Trainer configuration.
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        save_last=True,
        dirpath=cfg.paths.checkpoint_dir,
        filename="best"
    )
    val_eval_loss_callback = EvalCallback(dataloaders['val'], 'val', registry)
    ood_eval_loss_callback = EvalCallback(dataloaders['ood'], 'ood', registry)
    val_inference_callback = InferenceCallback(datasets['val'], 'val', registry, tokenizer, max_steps=cfg['train']['callbacks']['inference_max_steps'], 
                                               sample_size=cfg['train']['callbacks']['inference_sample_size'], resample_each_time=False)
    ood_inference_callback = InferenceCallback(datasets['ood'], 'ood', registry, tokenizer, max_steps=cfg['train']['callbacks']['inference_max_steps'], 
                                               sample_size=cfg['train']['callbacks']['inference_sample_size'], resample_each_time=False)

    trainer = L.Trainer(
        accelerator=cfg.general.accelerator,
        devices=cfg.general.devices,
        max_epochs=cfg.train.trainer.epochs,
        accumulate_grad_batches=cfg.train.trainer.accumulate_grad_batches,
        precision="16-mixed",
        val_check_interval=cfg['train']['trainer']['val_check_interval'],
        callbacks=[
            checkpoint_callback,
            val_eval_loss_callback,
            ood_eval_loss_callback,
            val_inference_callback,
            ood_inference_callback,
        ],
        logger=logger,
        log_every_n_steps=cfg['train']['trainer']['log_every_n_step']
    )

    trainer.fit(
        model, 
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val'],
    )


if __name__ == "__main__":
    main()
