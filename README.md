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
python3 ./scripts/generate_data.py --n_vars_range 5 15 --num_formulas 100000 --remap_variables 25 --output_file ./data/cdcl_data_train.json
```

- For training: 100k samples, variable range 5–15, remap to 25.
- For validation and test: same setup, but with 10k samples.
- For OOD eval: 5k samples, variable range 16–25.

### 2. Build the Tokenizer

```bash
python3 ./scripts/build_tokenizer.py
```

This creates the tokenizer files in `tokenizer/`.

### 3. Inspect the Dataset

```bash
python3 ./scripts/inspect_dataset.py
```

This shows tokenization behavior, label formatting, and generates length histograms. Useful for choosing a proper `block_size`.

### 4. Configure Training

Edit `config.yaml` to set model, training, and hardware parameters. For example:
- Set `block_size`, `batch_size`, etc.
- Set `general.devices` to match the cluster hardware.

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