from abc import ABC, abstractmethod
from typing import NamedTuple, Any, Dict, Union, List

class EnvInfo(NamedTuple):
    game_score: Any
    traj_done: bool

class EnvStep(NamedTuple):
    observation: Any  # Could be np.ndarray, Dict, etc.
    reward: Any
    done: bool
    env_info: Union[EnvInfo, Dict[str, Any]] 

class EnvSpaces(NamedTuple):
    observation: Any 
    action: Any 

class BaseEnv(ABC):
    name: str = "base_env"

    @classmethod
    def get_name(cls) -> str:
        return cls.name
    
    @abstractmethod
    def step(self, action: Any) -> EnvStep:
        """Run one timestep of the environment's dynamics."""
        pass
    
    @abstractmethod
    def reset(self) -> Any:
        """Resets the state of the environment and returns an initial observation."""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Any:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        pass
    
    @property
    def spaces(self) -> EnvSpaces:
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    @property
    @abstractmethod
    def horizon(self) -> int:
        pass

    def close(self) -> None:
        pass