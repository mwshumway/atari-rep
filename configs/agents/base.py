from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AgentConfig:
    """Generic agent configuration."""

    name: str = "rainbow"
    action_size: int = 6

    exploration_model: str = "online"  # online | target
    exploration_mode: Dict[str, str] = field(default_factory=lambda: {"backbone": "eval", "neck": "eval", "head": "eval"})

    rep: bool = False
    rep_candidate: Optional[str] = None

    double: bool = True
    v_min: float = -10.0
    v_max: float = 10.0

    aug_target: bool = False
    compile: bool = False
