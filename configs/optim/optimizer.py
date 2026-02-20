from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OptimizerConfig:
    """Optimizer configuration used by agents and pretraining."""

    type: str = "adam"  # adam | sgd | rmsprop
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.0

    kwargs: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "type": self.type,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "momentum": self.momentum,
        }
        if self.kwargs:
            base.update(self.kwargs)
        return base
