from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathsConfig:
    """Filesystem locations used by training and evaluation."""

    root_dir: Path = Path(".")
    data_dir: Path = Path("./data_storage/replay")
    log_dir: Path = Path("./data_storage/runs")
    ckpt_dir: Path = Path("./data_storage/checkpoints")
    buffer_dir: Path = Path("./data_storage/buffers")
    cache_dir: Path = Path("./.cache")
    output_dir: Path = Path("./outputs")
