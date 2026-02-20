from dataclasses import dataclass


@dataclass
class VecEnvConfig:
    """Vectorized environment settings."""

    num_envs: int = 1
    async_mode: bool = False
    reset_on_done: bool = True
