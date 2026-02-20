from dataclasses import dataclass, field
from typing import Optional

from configs.models.backbone import BackboneConfig
from configs.models.neck import NeckConfig
from configs.models.head import HeadConfig


@dataclass
class ModelConfig:
    """Full model configuration (backbone + neck + head)."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    head: HeadConfig = field(default_factory=HeadConfig)

    rep: bool = False
    rep_candidate: Optional[str] = None
