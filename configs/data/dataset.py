from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    """Dataset configuration for offline datasets and pretraining."""

    name: str = "default"
    root: str = "./data"
    split: str = "train"
    train_split: float = 0.98
    val_split: float = 0.02

    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    frame: int = 4
    frame_skip: int = 4
    imagesize: int = 84

    normalize: bool = False
    clip_reward: bool = True

    games: List[str] = field(default_factory=list)
    limit: Optional[int] = None
