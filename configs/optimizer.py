from dataclasses import dataclass, field

@dataclass
class OptimizerConfig:
    type: str = 'adam'
    lr: float = 0.0001 # sqrt(2)
    weight_decay: float = 0.0
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    eps: float =     0.00015