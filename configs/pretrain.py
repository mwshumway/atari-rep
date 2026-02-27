from dataclasses import dataclass


@dataclass
class PretrainConfig:
    type: str = "spr"
    num_epochs: int = 100

    use_amp: bool = True
    distributed: bool = False
    compile: bool = True

    clip_grad_norm: float = 0.5

    target_update_every: int = -1
    log_every: int = 1_000 # of steps
    save_every: int = 10  # of epochs
    eval_every: int = -1 # of epochs

    # ATC
    temperature: float = 1.0

    # CQL
    feature_normalization: bool = False
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    cql_coefficient: float = 0.1
    target_tau: float = 0.99

    checkpoint_dir: str = "data_storage/pretrained_models"
