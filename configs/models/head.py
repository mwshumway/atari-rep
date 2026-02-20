from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class HeadConfig:
    """Head configuration."""

    type: str = "mh_distributional"  # linear | mh_linear | mh_nonlinear | mh_distributional | mh_nonlinear_distributional | mae_head | siammae_head | spr_head | spr_idm_head | identity
    hidden_dim: int = 512
    out_dim: int = 512
    num_actions: int = 6
    num_atoms: int = 51
    num_heads: int = 1
    v_min: float = -10.0
    v_max: float = 10.0
    dropout: float = 0.0

    kwargs: Dict[str, Any] = field(default_factory=dict)
