from dataclasses import dataclass


@dataclass
class ATCPretrainConfig:
    """ATC pretraining configuration."""

    enabled: bool = False
    temperature: float = 0.1
    loss_weight: float = 1.0
    negative_samples: int = 64
    target_ema: float = 0.99
