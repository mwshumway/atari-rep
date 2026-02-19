"""
src/probes/__init__.py
======================
"""
from .datasets.action import ActionProbeDataset
from .datasets.reward import RewardProbeDataset
from .probes.action import ActionProbe
from .probes.reward import RewardProbe


def build_probe(cfg, device, model, dataloader):
    probe_type = cfg.probing.probe_type

    if probe_type == 'action':
        

        dataset = ActionProbeDataset(
            model, dataloader, device, cfg.probing.feature_extractor, cfg
        )
        probe = ActionProbe(cfg, device, cfg.model.action_size)
    elif probe_type == 'reward':
        dataset = RewardProbeDataset(
            model, dataloader, device, cfg.probingfeature_extractor, cfg
        )
        probe = RewardProbe(cfg, device)
    elif probe_type == 'value':
        from .probes.value import ValueProbe
        from .datasets.value import ValueProbeDataset

        dataset = ValueProbeDataset(
            model, dataloader, device, cfg.probing.feature_extractor, cfg
        )
        probe = ValueProbe(cfg, device)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    return probe, dataset