# SAT-LLM: Training Transformers on CDCL SAT Traces

This repository provides a full pipeline for training language models on traces from a CDCL SAT solver.

## 📁 Directory Overview

- `data/` — destination for generated datasets (`.json` format).
- `src/` — core source code including model, environment, and dataloaders.
- `scripts/` — Python entry-points such as `train.py`, `generate_data.py`, etc.
- `tokenizer/` — generated after building the tokenizer, stores `vocab.txt` and `tokenizer.json`.
- `runs/` — stores training checkpoints and logs.
- `requirements.txt` — lists all dependencies.

## ⚙️ Setup

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## 📌 Pipeline Instructions

### 1. Generate Datasets

Use the following script to create the training and evaluation datasets:

```bash
python3 ./scripts/generate_data.py --n_vars_range 5 15 --num_formulas 100000 --remap_vars_up_to --split train
```

- For training: 100k samples, variable range 5–15, remap to 25, seed to 42.
- For validation and test: same setup, but with 10k samples, seed to 53 and 64.
- For OOD eval: 5k samples, variable range 16–25, seed to 75.

For reproducibily, ensure the seed is set.

### 2. Build the Tokenizer

```bash
python3 ./scripts/build_tokenizer.py
```

This creates the tokenizer files in `tokenizer/`.

### 3. Preprocess the Dataset

```bash
python3 ./scripts/process_dataset.py  --inspect
```

This creates the .pt files with pretokenized data ready to be load during training. This '--inspect' flag shows tokenization behavior, label formatting, and generates length histograms. Useful for instance for choosing a proper `block_size`. It also creates checksums.

### 4. Configure Training

Edit `config.yaml` to set model, training, and hardware parameters. For example:
- Set `block_size`, `batch_size`, etc.
- Set `general.devices` to match the cluster hardware.
- Set `train.dataset.kind` to select which type of traces should the model be trained on.

### 5. Run Tests

Ensure everything is working before training:

```bash
pytest tests/
```

### 6. Start Training

```bash
python3 ./scripts/train.py
```

- Training logs are sent to Weights & Biases (WandB)
- Model checkpoints are saved in the `runs/` directory