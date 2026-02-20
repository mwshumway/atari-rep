from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from configs.common.paths import PathsConfig
from configs.common.runtime import RuntimeConfig
from configs.common.augmentation import AugmentationConfig
from configs.envs.atari import AtariEnvConfig
from configs.envs.vec_env import VecEnvConfig
from configs.data.dataset import DatasetConfig
from configs.models.model import ModelConfig
from configs.buffers.per import PERBufferConfig
from configs.optim.optimizer import OptimizerConfig
from configs.optim.scheduler import SchedulerConfig
from configs.agents.rainbow import RainbowConfig
from configs.training.train import TrainConfig
from configs.logging.logger import LoggerConfig
from configs.pretrain.base import PretrainConfig
from configs.probing.probing import ProbingConfig


@dataclass
class RootConfig:
    """Top-level configuration used by tyro CLIs."""

    task: str = "train"  # train | pretrain | probe | eval

    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    env: AtariEnvConfig = field(default_factory=AtariEnvConfig)
    vec_env: VecEnvConfig = field(default_factory=VecEnvConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    model: ModelConfig = field(default_factory=ModelConfig)
    buffer: PERBufferConfig = field(default_factory=PERBufferConfig)

    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig())
    prior_weight_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    eps_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    gamma_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
    n_step_scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())

    agent: RainbowConfig = field(default_factory=RainbowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    probing: ProbingConfig = field(default_factory=ProbingConfig)

    notes: Optional[str] = None
