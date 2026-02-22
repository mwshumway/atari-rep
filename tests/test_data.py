import pytest

def test_download_data(cfg):
    """Test that the data downloading function works correctly."""
    from src.data import download_data
    try:
        download_data(cfg)
    except Exception as e:
        assert False, f"download_data failed with error: {e}"

def test_build_dataloader(cfg):
    """Test that the dataloader building function works correctly."""
    from src.data import build_dataloader
    try:
        train_dataloader, train_sampler, eval_dataloader, eval_sampler = build_dataloader(cfg)
        assert train_dataloader is not None, "train_dataloader is None"
        assert eval_dataloader is not None, "eval_dataloader is None"
    except Exception as e:
        assert False, f"build_dataloader failed with error: {e}"


@pytest.fixture
def cfg():
    from configs.base import BaseConfig
    cfg = BaseConfig()
    cfg.games = ['amidar', 'atlantis', 'bank_heist', 'battle_zone', 'boxing', 
        'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber',
        'demon_attack', 'double_dunk', 'enduro', 'fishing_derby', 'freeway', 
        'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey',
        'jamesbond', 'kangaroo', 'krull', 'kung_fu_master', 'ms_pacman', 
        'name_this_game', 'phoenix', 'qbert', 'road_runner', 'robotank',
        'space_invaders', 'star_gunner', 'time_pilot', 'up_n_down', 'video_pinball',
        'wizard_of_wor', 'yars_revenge', 'zaxxon']
    return cfg


if __name__ == "__main__":
    test_download_data(cfg)
    print("✅ test_download_data passed successfully.")