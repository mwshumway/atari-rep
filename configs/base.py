import tyro
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .data import DataConfig
from .model import BackboneConfig, NeckConfig, HeadConfig, LoadModelConfig

@dataclass
class BaseConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    load_model: LoadModelConfig = field(default_factory=LoadModelConfig)

    games: List[str] = field(default_factory=lambda: [])

    num_gpus_per_node: int = 1

    device: str = "cuda" # "cuda" or "cpu"

    seed: int = 0

    frame: int = 4
    t_step: int = 1
    n_step: int = 3
    gamma: float = 0.99
    num_workers: int = 4 * num_gpus_per_node

    obs_shape: Tuple[int, int, int, int] = (4, 1, 84, 84)
    action_size: int = 18

    frame_skip: int = 4
    minimal_action_set: bool = True
    clip_reward: bool = True
    episodic_lives: bool = True
    max_start_noops=30
    repeat_action_probability=0.25
    horizon: int = 27_000
    stack_actions: int = 0
    grayscale: bool = True
    
