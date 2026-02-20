from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversionConfig:
    """TFRecord to numpy/torch conversion settings."""

    input_dir: str = "./data/tfrecords"
    output_dir: str = "./data/converted"
    game: Optional[str] = None
    max_episodes: Optional[int] = None
    overwrite: bool = False
