from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class NeckConfig:
    """Neck configuration."""

    type: str = "identity"  # identity | linear | mae_neck | mh_mlp | mh_spatial_mlp | siammae_neck | spatial_mlp | dt_neck
    hidden_dim: int = 512
    out_dim: int = 512
    num_heads: int = 1
    dropout: float = 0.0

    kwargs: Dict[str, Any] = field(default_factory=dict)
