from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SchedulerConfig:
    """Learning-rate or parameter scheduler configuration."""

    type: str = "linear"  # linear | exponential | cosine
    start: float = 1.0
    end: float = 0.0
    duration: int = 1_000_000
    warmup: int = 0
    gamma: float = 0.99

    kwargs: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "warmup": self.warmup,
            "gamma": self.gamma,
        }
        if self.kwargs:
            base.update(self.kwargs)
        return base
