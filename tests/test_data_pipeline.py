from src.dataset.raw_data_loader import RawDataLoader
from src.dataset.dataset_builder import DatasetBuilder
from src.dataset.dataloader_builder import DataloaderBuilder
from src.model.registry import CommandRegistry
from src.dataset.dataset import TokenizedDataset
import numpy as np


def test_dataloader(tmp_path, cfg, tokenizer, tokenized_dataset):
    cfg_cp = cfg.copy()
    registry = CommandRegistry(cfg_cp, tokenizer)
    dataset = tokenized_dataset

    # --- Save + load ---
    save_path = tmp_path / "tokenized.pt"
    dataset.save(save_path)
    reloaded = TokenizedDataset.load(save_path)
    assert len(reloaded) == len(dataset)

    # --- Build dataloader ---
    loader_builder = DataloaderBuilder(
        batch_size=2,
        registry=registry,
        num_workers=0
    )
    loader = loader_builder.build_dataloader(reloaded, shuffle=False)
    batch = next(iter(loader))

    # --- Validate batch structure ---
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

def test_curriculum_schedule_behavior():
    max_epochs = 20
    n_data = 1_000_000
    min_length = 80
    max_length = 1500
    temp = 1.2

    def generate_right_skewed_ints(N: int, min_val: int, max_val: int, scale: float) -> list[int]:
        samples = np.random.exponential(scale=scale, size=int(N * 1.2))
        scaled = samples + min_val
        ints = scaled.astype(int)
        valid = ints[(ints >= min_val) & (ints < max_val)]
        return valid[:N].tolist()

    lengths = generate_right_skewed_ints(n_data, min_val=min_length, max_val=max_length, scale=300.0)

    for epoch in range(max_epochs):
        weights = TokenizedDataset.compute_weights(epoch, max_epochs, lengths, temp=temp)
        weighted_mean = (weights * lengths).sum()  # weights is a prob distribution
        print(f'Epoch {epoch}: weighted_mean={weighted_mean}')

    # Sort to find extremes
    lengths.sort()
    extremes_dataset = [min_length, (max_length - min_length) // 2 + min_length, max_length]
    first_epoch, middle_epoch, last_epoch = 0, (max_epochs - 1) // 2, max_epochs - 1
    weights_early = TokenizedDataset.compute_weights(first_epoch, max_epochs, extremes_dataset, temp=temp)
    weights_mid = TokenizedDataset.compute_weights(middle_epoch, max_epochs, extremes_dataset, temp=temp)
    weights_late = TokenizedDataset.compute_weights(last_epoch, max_epochs, extremes_dataset, temp=temp)

    print(f"Epoch {first_epoch}: short={weights_early[0]:.5f}, middle={weights_early[1]:.5f}, long={weights_early[2]:.5f}")
    print(f"Epoch {middle_epoch}: short={weights_mid[0]:.5f}, middle={weights_mid[1]:.5f}, long={weights_mid[2]:.5f}")
    print(f"Epoch {last_epoch}: short={weights_late[0]:.5f}, middle={weights_late[1]:.5f}, long={weights_late[2]:.5f}\n")

    # --- Assertions ---
    assert weights_early[0] > weights_early[1] > weights_early[2], "Short traces should be favored early"
    assert weights_late[2] > weights_late[1] > weights_late[0], "Long traces should be favored late"
