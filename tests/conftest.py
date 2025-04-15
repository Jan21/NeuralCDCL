import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
import torch

import pytest
import copy
from tokenizers import Tokenizer
from omegaconf import OmegaConf
from src.model.registry import CommandRegistry


def pytest_configure(config):
    seed = int(config.getoption("seed"))
    print(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------------------------------------------------
# CLI option: --tokenizer-path (relative to project root)
# -------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--tokenizer-path",
        action="store",
        default="tokenizer/tokenizer.json",
        help="Path to tokenizer.json file (relative to project root)"
    )
    parser.addoption(
        "--config-path",
        action="store",
        default="config/config.yaml",
        help="Path to real OmegaConf config.yaml (relative to project root)"
    )
    parser.addoption(
        "--tiny-data-path",
        action="store",
        default="tests/testdata/cdcl_data_tiny.json",
        help="Path to the small cdcl data used for testing (relative to project root)"
    )
    parser.addoption(
        "--seed",
        action="store",
        default="42",
        help="Random seed for reproducibility"
    )

# -------------------------------------------------------------------
# Fixture: real tokenizer (loaded from file via CLI option)
# -------------------------------------------------------------------
@pytest.fixture(scope="session")
def tokenizer(pytestconfig):
    relative_path = pytestconfig.getoption("tokenizer_path")
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relative_path))
    assert os.path.exists(abs_path), f"Tokenizer not found: {abs_path}"
    return Tokenizer.from_file(abs_path)

# -------------------------------------------------------------------
# Fixture: real config (OmegaConf loaded from YAML)
# -------------------------------------------------------------------
@pytest.fixture(scope="session")
def cfg(pytestconfig):
    relative_path = pytestconfig.getoption("config_path")
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relative_path))
    assert os.path.exists(abs_path), f"Config file not found: {abs_path}"
    return OmegaConf.load(abs_path)

# -------------------------------------------------------------------
# Fixture: real tokenized samples
# -------------------------------------------------------------------
@pytest.fixture
def tokenized_dataset(pytestconfig, cfg, tokenizer):
    from src.dataset.pipeline import DatasetPipeline 

    data_relative_path = pytestconfig.getoption("tiny_data_path")
    data_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", data_relative_path))
    cfg_cp = copy.deepcopy(cfg)
    cfg_cp["data"]["files"] = {'train': data_abs_path}
    cfg_cp["data"]["num_workers"] = 1
    cfg_cp["data"]["tokenize_batch_size"] = 16
    registry = CommandRegistry(cfg, tokenizer)
    pipeline = DatasetPipeline(cfg_cp, tokenizer, registry)
    datasets = pipeline.build(filter_by_len=True)

    train_set = datasets["train"]
    assert len(train_set) > 0, "Train set is empty!"

    return train_set
