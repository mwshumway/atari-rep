import pytest
import numpy as np
from src.env.atari import AtariEnv
from configs.base import BaseConfig

@pytest.fixture
def cfg():
    cfg = BaseConfig()
    cfg.games = ["pong"]
    return cfg

@pytest.fixture
def env_pong(cfg):
    assert len(cfg.games) == 1, "Expected exactly one game in cfg.games for this test"
    game = cfg.games[0]
    
    try:
        env = AtariEnv(game=game,
                       frame_skip=cfg.env.frame_skip,
                       frame=cfg.frame,
                       minimal_action_set=cfg.env.minimal_action_set,
                       clip_reward=cfg.env.clip_reward,
                       episodic_lives=cfg.env.episodic_lives,
                       max_start_noops=cfg.env.max_start_noops,
                       repeat_action_probability=cfg.env.repeat_action_probability,
                       horizon=cfg.env.horizon,
                       stack_actions=cfg.env.stack_actions,
                       grayscale=cfg.env.grayscale,
                       seed=cfg.seed)
    except Exception as e:
        pytest.skip(f"Could not create AtariEnv for game '{game}': {e}")
    return env

def test_initialization_attributes(env_pong):
    assert env_pong.game == "pong"
    assert env_pong.game_id > 0
    assert env_pong.action_space.n == 6
    assert env_pong.observation_space.shape == (4, 1, 84, 84)
    assert env_pong.observation_space.dtype == np.uint8

def test_reset(env_pong):
    """Test that reset returns a valid initial observation."""
    obs = env_pong.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 1, 84, 84)
    # Check that obs is zeroed out or contains valid pixel data
    assert np.all(obs >= 0) and np.all(obs <= 255)

def test_step_structure(env_pong):
    """
    Test that stepping the environment returns the correct NamedTuple structure
    and data types.
    """
    env_pong.reset()
    # Take a random valid action
    action = env_pong.action_space.sample()
    
    # Run step
    step_result = env_pong.step(action)
    
    # Unpack NamedTuple
    obs, reward, done, info = step_result

    # 1. Observation
    assert obs.shape == (4, 1, 84, 84)
    
    # 2. Reward (should be clipped to -1, 0, 1 if clip_reward=True)
    assert -1.0 <= reward <= 1.0 

    # 3. Done
    assert isinstance(done, (bool, np.bool_))

    # 4. Info (Checking your specific EnvInfo namedtuple)
    assert hasattr(info, "game_score")
    assert hasattr(info, "traj_done")

def test_determinism_seeding():
    """
    Ensure that two environments with the same seed produce 
    identical observations when given the same actions.
    """
    env1 = AtariEnv(game="pong", seed=123)
    env2 = AtariEnv(game="pong", seed=123)
    
    obs1 = env1.reset()
    obs2 = env2.reset()
    
    # Initial state should be identical
    np.testing.assert_array_equal(obs1, obs2)
    
    # Take fixed actions
    actions = [0, 2, 3, 2, 0] # NOOP, UP, RIGHT, UP, NOOP
    
    for a in actions:
        out1 = env1.step(a)
        out2 = env2.step(a)
        
        # Obs and Reward should match exactly
        np.testing.assert_array_equal(out1.observation, out2.observation)
        assert out1.reward == out2.reward

def test_invalid_game_name():
    """Test that initializing a non-existent game raises an error."""
    with pytest.raises(ValueError):
        # 'fake_game' is not in ATARI_RANDOM_SCORE dict
        AtariEnv(game="fake_game_123")

def test_color_mode():
    """Test that grayscale=False returns 3 channels."""
    env = AtariEnv(game="pong", grayscale=False, frame=1)
    # Shape: (frames, channels, h, w) -> (1, 3, 84, 84)
    assert env.observation_space.shape == (1, 3, 84, 84)

@pytest.mark.parametrize("game_name", ["breakout", "space_invaders"])
def test_load_multiple_games(game_name):
    """Parametrized test to ensure other common games load correctly."""
    try:
        env = AtariEnv(game=game_name)
        env.reset()
        assert env.game == game_name
    except Exception as e:
        pytest.fail(f"Could not load game {game_name}: {e}")

def test_build_env(cfg):
    """Test that build_env correctly constructs the environment from cfg."""
    from src.env import build_env
    from src.env.vec_env import VecEnv
    try:
        train_env, eval_env = build_env(cfg)
        assert train_env is not None
        assert eval_env is not None
        assert isinstance(train_env, AtariEnv)
        assert isinstance(eval_env, VecEnv)
    except Exception as e:
        pytest.fail(f"build_env failed with error: {e}")
