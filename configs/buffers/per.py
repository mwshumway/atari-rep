from dataclasses import dataclass
from typing import Tuple


@dataclass
class PERBufferConfig:
    """Prioritized Experience Replay buffer settings."""

    name: str = "per_buffer"
    size: int = 1_000_000
    prior_exp: float = 0.6
    max_n_step: int = 10
    save_backbone_feat: bool = False
    backbone_feat_shape: Tuple[int, ...] = (1, 512)
