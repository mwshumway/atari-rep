from dataclasses import dataclass

@dataclass
class EnvConfig:
    type: str = "atari"
    num_train_envs: int = 1
    num_eval_envs: int = 100

    frame_skip: int = 4
    minimal_action_set: bool = True
    clip_reward: bool = True
    episodic_lives: bool = True
    max_start_noops: int = 30
    repeat_action_probability: float = 0.25
    horizon: int = 27_000
    stack_actions: int = 0
    grayscale: bool = True    


@dataclass
class EvalEnvConfig:
    type: str = "atari"

    frame_skip: int = 4
    minimal_action_set: bool = True
    clip_reward: bool = True
    episodic_lives: bool = False
    max_start_noops: int = 30
    repeat_action_probability: float = 0.0
    horizon: int = 27_000
    stack_actions: int = 0
    grayscale: bool = True