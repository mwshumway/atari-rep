"""
Note: max_step is set in parent/base config
"""

from dataclasses import dataclass

@dataclass
class PriorWeightSchedulerConfig:
    type: str = "linear"
    initial_value: float = 0.4
    final_value: float = 1.0
    max_step: int = 0


@dataclass
class EpsSchedulerConfig:
    type: str = "linear"
    initial_value: float = 1.0
    final_value: float = 0.01
    max_step: int = 0

@dataclass
class GammaSchedulerConfig:
    type: str = "exponential"
    initial_value: float = 0.99
    final_value: float = 0.99
    reverse: bool = True
    max_step: int = 0

@dataclass
class NStepSchedulerConfig:
    type: str = "exponential"
    initial_value: int = 10
    final_value: int = 10
    reverse: bool = False
    max_step: int = 0
    