from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoggerConfig:
    """Logging configuration (wandb + local summaries)."""

    use_wandb: bool = True
    project: str = "atari-rep"
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    log_interval: int = 100
    eval_interval: int = 10_000
    save_interval: int = 50_000

    save_model: bool = True
    save_buffer: bool = True
    save_videos: bool = False
    video_interval: int = 50_000
