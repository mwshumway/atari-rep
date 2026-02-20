from dataclasses import dataclass


@dataclass
class SPRPretrainConfig:
    """SPR pretraining configuration."""

    enabled: bool = True
    prediction_horizon: int = 10
    loss_weight: float = 1.0
    target_ema: float = 0.99
    k_step: int = 1
    lambda_reg: float = 0.0
