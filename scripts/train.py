import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
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
from src.dataset.pipeline import DatasetPipeline
from src.model.registry import CommandRegistry
from src.model.callbacks.eval import EvalCallback
from src.model.callbacks.inference import InferenceCallback
from src.model.lit_wrapper import LitWrapper


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Save config.
    config_path = to_absolute_path(cfg.paths.config_export)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    OmegaConf.save(cfg, config_path)

    # For reproducibility.
    random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)

    # Tokenizer.
    tokenizer = Tokenizer.from_file(to_absolute_path(cfg.paths.tokenizer))
    registry = CommandRegistry(cfg, tokenizer)

    # Data.
    pipeline = DatasetPipeline(cfg, tokenizer, registry)
    datasets = pipeline.build()
    dataloaders = pipeline.build_dataloaders(datasets)

    # LitGPT.
    lit_cfg = Config(
        name="cdcl-pythia",
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
        val_check_interval=cfg['train']['callbacks']['val_check_interval'],
        callbacks=[
            checkpoint_callback,
            val_eval_loss_callback,
            ood_eval_loss_callback,
            val_inference_callback,
            ood_inference_callback,
        ],
        logger=logger,
        log_every_n_steps=cfg['train']['callbacks']['train_log_every_n_steps']
    )

    trainer.fit(
        model, 
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val'],
    )


if __name__ == "__main__":
    main()
