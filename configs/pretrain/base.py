from dataclasses import dataclass, field

from configs.pretrain.spr import SPRPretrainConfig
from configs.pretrain.atc import ATCPretrainConfig
from configs.pretrain.cql import CQLPretrainConfig


@dataclass
class PretrainConfig:
    """Pretraining options and algorithm-specific configs."""

    enabled: bool = False
    name: str = "spr"  # spr | atc | cql

    spr: SPRPretrainConfig = field(default_factory=SPRPretrainConfig)
    atc: ATCPretrainConfig = field(default_factory=ATCPretrainConfig)
    cql: CQLPretrainConfig = field(default_factory=CQLPretrainConfig)
