from dataclasses import dataclass

from configs.agents.base import AgentConfig


@dataclass
class RainbowConfig(AgentConfig):
    """Rainbow-specific agent configuration."""

    name: str = "rainbow"
    n_step: int = 10
    gamma: float = 0.99
    atom_size: int = 51
