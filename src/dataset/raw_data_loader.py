import json
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pathlib import Path


class RawDataLoader:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

    def load_all(self) -> dict[str, list[dict]]:
        result = {}
        for split in self._cfg.data.files.keys():
            result[split] = self.load(split)
        return result

    def load(self, split: str) -> list[dict]:
        abs_path = Path(to_absolute_path(self._cfg.data.raw_files[split]))
        with open(abs_path, "r") as f:
            result = json.load(f)
        return result
        