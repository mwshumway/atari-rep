from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeConfig:
    """Runtime and hardware settings."""

    seed: int = 42
    device: str = "cuda"  # cuda | cpu | mps
    compile: bool = False
    deterministic: bool = True
    cudnn_benchmark: bool = True

    num_workers: int = 4
    pin_memory: bool = True

    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    precision: str = "fp32"  # fp32 | fp16 | bf16
    anomaly_detection: bool = False
    profile: bool = False
    log_gpu_mem: bool = False

    resume: bool = False
    ckpt_path: Optional[str] = None
