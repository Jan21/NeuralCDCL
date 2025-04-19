import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from src.model.registry import CommandRegistry
from src.model.callbacks.eval import EvalCallback
from lightning.pytorch.core.module import LightningModule


class DummyModel(LightningModule):
    def forward(self, x):
        # Always predict token 1 for all positions
        B, T = x.shape
        V = 10
        logits = torch.zeros(B, T, V)
        logits[:, :, 1] = 10.0
        return logits


def test_eval_callback_basic(cfg, tokenizer):
    registry = CommandRegistry(cfg, tokenizer)

    # Set up dummy trace type tokens
    registry.solve_block_markers = [99]
    registry.up_block_markers = [77]
    registry.ac_block_markers = [55]

    # Prepare dummy batch: one solve, one up, one ac
    B = 3
    T = 5
    # Individual examples as dicts
    dataset = [
        {
            "input_ids": torch.tensor([99, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, 1, 1, 1, 1]),  # solve
        },
        {
            "input_ids": torch.tensor([77, 1, 2, 1, 1]),
            "labels": torch.tensor([-100, 1, 2, -100, -100]),  # up
        },
        {
            "input_ids": torch.tensor([55, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, 2, 2, 2, 2]),  # ac
        },
    ]

    # Collate into batch (you could use your actual collate_fn if you have one)
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    dummy_loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)

    callback = EvalCallback(loader=dummy_loader, loader_name="test", registry=registry, sample_size=None)

    # Simulate trainer and trainer.logger
    trainer = SimpleNamespace()
    trainer.current_epoch = 0
    trainer.logger = SimpleNamespace()
    trainer.logger.experiment = SimpleNamespace()
    trainer.logger.experiment.log = lambda *args, **kwargs: None  # no-op

    model = DummyModel()

    # Patch log_dict method to capture metrics
    logged = {}

    def fake_log_dict(dct, **_):
        for k, v in dct.items():
            logged[k] = v.item() if torch.is_tensor(v) else v

    model.log_dict = fake_log_dict

    callback.on_train_epoch_end(trainer, model)

    # Check expected keys and values
    assert "test/solve/token_accuracy" in logged
    assert "test/solve/full_sequence_accuracy" in logged
    assert "test/solve/result_match_accuracy" in logged
    assert "test/up/token_accuracy" in logged
    assert "test/ac/token_accuracy" in logged

    # solve trace is all 1s → correct
    assert logged["test/solve/token_accuracy"] == 1.0
    assert logged["test/solve/full_sequence_accuracy"] == 1.0
    assert logged["test/solve/result_match_accuracy"] == 1.0

    # up trace has masking and is correct only on 3 positions
    assert logged["test/up/token_accuracy"] == 0.5
    assert logged["test/up/full_sequence_accuracy"] == 0.0

    # ac trace: model predicts 1, label is 2 → all wrong
    assert logged["test/ac/token_accuracy"] == 0.0
