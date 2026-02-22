import pytest
import torch

@pytest.fixture
def cfg():
    from configs.base import BaseConfig
    cfg = BaseConfig()
    return cfg

def test_model_initialization(cfg):
    from src.model import build_model
    device = torch.device("cpu")
    try:
        model = build_model(cfg, device)
        assert model is not None, "Model initialization failed, got None"
    except Exception as e:
        assert False, f"Model initialization failed with error: {e}"


def test_load_pretrained(cfg):
    from src.model import build_model
    from src.model.model_utils import load_model_weights
    device = torch.device("cpu")
    model = build_model(cfg, device)
    checkpoint_path = "/projectnb/ds598xz/students/mshumway/atari-rep-bench/materials/model/pretrain_cql/cql/cql-dist_resnet/epoch90.pth"
    try:
        load_model_weights(model, checkpoint_path, device, load_layers=cfg.load_model.load_layers)
    except Exception as e:
        assert False, f"Loading pretrained model failed with error: {e}"


if __name__ == "__main__":
    test_model_initialization(cfg)
    test_load_pretrained(cfg)