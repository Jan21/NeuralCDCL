from src.dataset.raw_data_loader import RawDataLoader
from src.dataset.dataset_builder import DatasetBuilder
from src.dataset.dataloader_builder import DataloaderBuilder
from src.model.registry import CommandRegistry
from src.dataset.dataset import TokenizedDataset


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
