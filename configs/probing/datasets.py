from dataclasses import dataclass
from typing import Optional


@dataclass
class ProbingDatasetConfig:
    """Probing dataset settings."""

    name: str = "action"
    root: str = "./data"
    split: str = "train"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    limit: Optional[int] = None
