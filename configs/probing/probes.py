from dataclasses import dataclass


@dataclass
class ProbingProbeConfig:
    """Probe head configuration for linear probing."""

    name: str = "linear"
    hidden_dim: int = 512
    out_dim: int = 18
    dropout: float = 0.0
