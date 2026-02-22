from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BackboneConfig:
    """Configuration for model backbone."""
    type: str = "resnet"
    in_shape: Optional[tuple] = None
    action_size: Optional[int] = None
    net_type: str = "resnet50"
    norm_type: str = "gn"
    width_multiplier: float = 1.0


@dataclass
class NeckConfig:
    """Configuration for model neck."""
    type: str = "mh_mlp"
    in_shape: Optional[tuple] = None
    action_size: Optional[int] = None
    hidden_dims: tuple = (1024, 512)
    norm_type: str = "ln"
    num_heads: int = 51


@dataclass
class HeadConfig:
    """Configuration for model head."""
    type: str = "mh_nonlinear_distributional"
    in_shape: Optional[tuple] = None
    action_size: Optional[int] = None
    hidden_sizes: tuple = (128, 128, 64)
    num_heads: int = 51
    num_atoms: int = 51


@dataclass
class LoadModelConfig:
    """Configuration for loading a pretrained model."""
    enable: bool = False
    model_path: str = ""
    load_layers: tuple = ("backbone", "neck")


