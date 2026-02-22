import tyro
from dataclasses import dataclass, field
from typing import Optional, List

from .data import DataConfig

@dataclass
class BaseConfig:
    data: DataConfig = field(default_factory=DataConfig)

    games: List[str] = field(default_factory=lambda: [])

    num_gpus_per_node: int = 1

    frame: int = 4
    t_step: int = 1
    n_step: int = 3
    gamma: float = 0.99
    num_workers: int = 4 * num_gpus_per_node