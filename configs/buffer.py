from dataclasses import dataclass

@dataclass
class BufferConfig:
    type: str = "per_buffer"
    size: int = 100_000
    prior_exp: float = 0.5
    max_n_step: int = 10
    