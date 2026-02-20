from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AugmentationConfig:
    """Image augmentation settings used by replay sampling and pretraining."""

    aug_types: List[str] = field(default_factory=lambda: ["random_shift"])
    mask_ratio: Optional[float] = None
    apply_to_target: bool = False
