from dataclasses import dataclass
from typing import Tuple


@dataclass
class ProbeConfig:
    test_frac: float = 0.2
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 500
    n_step: int = 10
    patience: int = 50
    min_delta: float = 1e-4

    hidden_sizes: Tuple[int, ...] = ()

    dataset_size: int = 25_000

    # Configs for probing experiments
    type: str = "offline" # "online" or "offline"
    analyze_only: bool = False
    policy_ckpt_dir: str = ""
    save_dir: str = "data_storage/probe_results"