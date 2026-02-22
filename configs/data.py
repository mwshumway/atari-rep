from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DataConfig:
    """Configuration for data collection and loading."""
    dataset_name: str = "pretrain"  # pretrain | near_ood_ft | etc.
    dataset_class: str = "default"  # default \ etc.
    data_dir: str = "data_storage/replay"

    runs: List[int] = field(default_factory=lambda: [1, 2])
    checkpoints: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    samples_per_checkpoint: int = 10_000

    eval_ratio: float = 0.01 # [0, 1]
    batch_size: int = 512

    shuffle: bool = True # whether to shuffle data during traning
    distributed: bool = False # whether to use distributed samplers for dataloader
    prefetch_factor: int = 2
    pin_memory: bool = True
