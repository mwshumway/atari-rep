from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BackboneConfig:
    """Backbone configuration."""

    type: str = "nature"  # nature | impala | resnet | identity
    in_channels: int = 4
    feature_dim: int = 512
    use_bn: bool = False
    dropout: float = 0.0

    kwargs: Dict[str, Any] = field(default_factory=dict)
