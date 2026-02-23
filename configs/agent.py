from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    type: str = "rainbow"
    num_timesteps: int = 100_000

    min_buffer_size: int = 2_000
    v_min: float = -10.0
    v_max: float = 10.0

    compile: bool = True
    rep: bool = False
    rep_candidate: str = ""

    aug_types: list = field(default_factory=lambda: ["random_shift", "intensity"])
    aug_target: bool = True

    double: bool = True
    optimize_per_env_step: int = 2
    reset_per_optimize_step: int = -1
    batch_size: int = 32
    clip_grad_norm: float = 10.0

    target_tau: float = 0.99

    eval_freq: int = -1
    rollout_freq: int = 10_000
    probe_on_policy_freq: int = -1
    save_freq: int = -1
    log_freq: int = 1_000

    max_rollout_steps: int = 10_000
    eval_eps: float = 0.001

    exploration_model: str = "target"
    update_buffer: bool = True