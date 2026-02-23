import tyro
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .data import DataConfig
from .model import BackboneConfig, NeckConfig, HeadConfig, LoadModelConfig
from .env import EnvConfig
from .buffer import BufferConfig
from .scheduler import PriorWeightSchedulerConfig, EpsSchedulerConfig, GammaSchedulerConfig, NStepSchedulerConfig
from .optimizer import OptimizerConfig
from .agent import AgentConfig
from .wandb import WandbConfig

@dataclass
class BaseConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    load_model: LoadModelConfig = field(default_factory=LoadModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    prior_weight_scheduler: PriorWeightSchedulerConfig = field(default_factory=PriorWeightSchedulerConfig)
    eps_scheduler: EpsSchedulerConfig = field(default_factory=EpsSchedulerConfig)
    gamma_scheduler: GammaSchedulerConfig = field(default_factory=GammaSchedulerConfig)
    n_step_scheduler: NStepSchedulerConfig = field(default_factory=NStepSchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    games: List[str] = field(default_factory=lambda: [])

    num_gpus_per_node: int = 1

    device: str = "cuda" # "cuda" or "cpu"

    seed: int = 0

    frame: int = 4
    t_step: int = 1
    n_step: int = 3
    gamma: float = 0.99
    num_workers: int = 4 * num_gpus_per_node

    obs_shape: Tuple[int, int, int, int] = (4, 1, 84, 84)
    action_size: int = 18

    env_type: str = "atari"
    num_train_envs: int = 1
    num_eval_envs: int = 100

    
