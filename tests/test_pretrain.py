import pytest
import torch
import copy

from src.pretrain import build_trainer

@pytest.fixture
def cfg():
    from configs import BaseConfig
    cfg = BaseConfig()
    cfg.pretrain.num_epochs = 1
    cfg.pretrain.compile = False

    cfg.games = ['amidar', 'atlantis', 'bank_heist', 'battle_zone', 'boxing', 
        'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber',
        'demon_attack', 'double_dunk', 'enduro', 'fishing_derby', 'freeway', 
        'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey',
        'jamesbond', 'kangaroo', 'krull', 'kung_fu_master', 'ms_pacman', 
        'name_this_game', 'phoenix', 'qbert', 'road_runner', 'robotank',
        'space_invaders', 'star_gunner', 'time_pilot', 'up_n_down', 'video_pinball',
        'wizard_of_wor', 'yars_revenge', 'zaxxon']

    return cfg

def test_load_trainer(cfg):
    atc_cfg = copy.deepcopy(cfg)
    atc_cfg.pretrain.type = "atc"
    spr_cfg = copy.deepcopy(cfg)
    spr_cfg.pretrain.type = "spr"
    cql_cfg = copy.deepcopy(cfg)
    cql_cfg.pretrain.type = "cql"


    device = torch.device("cpu")

    from src.model import build_model
    model = build_model(cfg, device)
    from src.logger import PretrainLogger
    logger = PretrainLogger(cfg)
    from src.data import build_dataloader
    train_dataloader, train_sampler, eval_dataloader, eval_sampler = build_dataloader(cfg)

    atc_trainer = build_trainer(atc_cfg, device, train_dataloader, train_sampler, eval_dataloader, eval_sampler, logger, model)
    spr_trainer = build_trainer(spr_cfg, device, train_dataloader, train_sampler, eval_dataloader, eval_sampler, logger, model)
    cql_trainer = build_trainer(cql_cfg, device, train_dataloader, train_sampler, eval_dataloader, eval_sampler, logger, model)

    assert atc_trainer is not None
    assert spr_trainer is not None
    assert cql_trainer is not None
