from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training loop configuration."""

    num_timesteps: int = 5_000_000
    min_buffer_size: int = 80_000
    optimize_per_env_step: int = 1
    batch_size: int = 32
    target_update_interval: int = 8_000

    eval_episodes: int = 10
    eval_interval: int = 50_000
    checkpoint_interval: int = 100_000

    gradient_clip_norm: float = 10.0
    log_histograms: bool = False
