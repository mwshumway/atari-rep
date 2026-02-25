from dataclasses import dataclass


@dataclass
class ProbeConfig:
    test_frac: float = 0.2
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 500
    n_step: int = 10
    patience: int = 50
    min_delta: float = 1e-4

    hidden_sizes: tuple= ()