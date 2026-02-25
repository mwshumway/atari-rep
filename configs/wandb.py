from dataclasses import dataclass
from typing import Optional

@dataclass
class WandbConfig:
    project: str = "debugging"
    entity: str = "mshumway-boston-university"
    group: str = "default"
    name: Optional[str] = None
    run_id: Optional[str] = None
    enabled: bool = False