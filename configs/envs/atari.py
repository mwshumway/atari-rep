from dataclasses import dataclass


@dataclass
class AtariEnvConfig:
    """Configuration for AtariEnv."""

    game: str = "pong"
    frame_skip: int = 4
    frame: int = 4
    minimal_action_set: bool = True
    clip_reward: bool = True
    episodic_lives: bool = True
    max_start_noops: int = 30
    repeat_action_probability: float = 0.0
    horizon: int = 9000
    stack_actions: int = 0
    grayscale: bool = True
    imagesize: int = 84
    seed: int = 42
    id: int = 0
