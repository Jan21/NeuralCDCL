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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import random
from tokenizers import Tokenizer
from src.data.pipeline import DatasetPipeline
from src.model.registry import CommandRegistry
from src.model.callbacks.eval_loss import EvalLossCallback
from src.model.callbacks.inference import InferenceCallback
from src.model.lit_wrapper import LitWrapper


# os.environ["WANDB_MODE"] = "disabled"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # For reproducibility.
    random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)

    # Tokenizer.
    tokenizer = Tokenizer.from_file(to_absolute_path(cfg.paths.tokenizer))
    registry = CommandRegistry(cfg, tokenizer)

    # Data.
    pipeline = DatasetPipeline(cfg, tokenizer)
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
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
        dirpath=cfg.paths.checkpoint_dir,
        filename="best"
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    # ood_eval_loss_callback = EvalLossCallback(dataloaders['ood'], 'ood', F.cross_entropy, eval_every_n_steps=100)
    inference_val_callback = InferenceCallback(datasets['val'], 'val', registry, tokenizer, max_steps=1000, sample_count=20, resample_each_time=False, seed=42)
    # inference_ood_callback = InferenceCallback(datasets['ood'], 'ood', registry, tokenizer, max_steps=500, sample_count=20, resample_each_time=False, seed=42)

    trainer = L.Trainer(
        # accelerator="cuda",
        # devices=cfg.general.devices,
        accelerator="cpu",
        devices=1,
        max_epochs=cfg.train.trainer.epochs,
        accumulate_grad_batches=cfg.train.trainer.accumulate_grad_batches,
        precision="16-mixed",
        val_check_interval=100,
        callbacks=[
            checkpoint_callback,
            lr_monitor_callback,
            # ood_eval_loss_callback,
            inference_val_callback,
            # inference_ood_callback
        ],
        logger=logger,
        log_every_n_steps=10
    )

    trainer.fit(
        model, 
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val'],
    )


if __name__ == "__main__":
    main()
