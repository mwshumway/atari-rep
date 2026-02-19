"""
src/data/data_utils.py
======================

A lot of this code is adapted from the Atari-PT codebase:
https://github.com/dojeon-ai/Atari-PB
"""

import os
import numpy as np
import h5py
from pathlib import Path
from copy import deepcopy
import gzip
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tqdm

from src.data.default_dataset import DefaultDataset

FILE_TYPES = ['observation', 'action', 'reward', 'terminal', 'game_id', 'rtg']

def data_downloaded(cfg):
    """
    Check if the data specified in the configuration is downloaded.
    """

    dataset_type = cfg.dataset_type

    # check in cfg.data_dir/dataset_type for files
    dataset_loc = os.path.join(cfg.data_dir, dataset_type)  # e.g. atari-rep-bench/data/pretrain/
    if not os.path.exists(dataset_loc):
        return False
    
    # Compute expected number of files
    # We expect 4 files per checkpoint, for each game and run
    n_expected_files = len(cfg.games) * len(cfg.runs) * len(cfg.checkpoints) * 4

    num_files = sum(1 for _ in Path(dataset_loc).rglob("*") if _.is_file())
    if num_files >= n_expected_files:
        return True
    else:
        print(f"len(files) = {num_files} vs. expected files: {n_expected_files}")
        return False
    

def hdf5_dataset_exists(file_paths):
    """
    Check if the consolidated HDF5 dataset files exist.

    IMPORTANT: This only checks for file existence, not necessarily
    that they are full of the desired data...

    :param file_paths: Dictionary of expected HDF5 file paths
    :return: True if all files exist, False otherwise
    """
    for file_type, file_path in file_paths.items():
        if not file_path.exists():
            return False
    return True


def get_hdf5_files(cfg_dict):
    """
    Gets a dict of file paths for the consolidated HDF5 files.
    """
    file_paths = {}
    for file_type in FILE_TYPES:
        if file_type == 'observation':
            file_path = Path(f"{cfg_dict['data_dir']}/consolidated/{cfg_dict['dataset_type']}/{file_type}.hdf5")
        else:
            file_path = Path(f"{cfg_dict['data_dir']}/consolidated/{cfg_dict['dataset_type']}/{file_type}_{cfg_dict['probe_n_step']}_{cfg_dict['gamma']}.hdf5")
        file_paths[file_type] = file_path
    
    return file_paths


def build_hdf5_dataset(cfg_dict):
    """
    Consolidates all individual .gz files into a single HDF5 file per data type,
    organized as [game_idx, run_idx, ckpt_idx, sample_idx].
    """
    file_paths = get_hdf5_files(cfg_dict)
    print(file_paths)

    if hdf5_dataset_exists(file_paths):
        print("HDF5 dataset already exists. Skipping consolidation.")
        return file_paths

    print("Creating consolidated HDF5 dataset files...")

    # Temporary paths
    tmp_paths = {ftype: path.with_suffix('.hdf5.tmp') for ftype, path in file_paths.items()}

    obs_exist = os.path.exists(file_paths['observation'])
    file_dict = open_hdf5_files(tmp_paths, obs_exist)

    n_games, n_runs, n_checkpoints = len(cfg_dict['games']), len(cfg_dict['runs']), len(cfg_dict['checkpoints'])

    try:
        print("=" * 50)
        print(cfg_dict['games'])
        print("=" * 50)
        for game_idx, game in enumerate(cfg_dict['games']):
            for run_idx, run in enumerate(cfg_dict['runs']):
                for ckpt_idx, ckpt in enumerate(cfg_dict['checkpoints']):
                    dataset = load_checkpoint_dataset(cfg_dict, game, run, ckpt, obs_exist)

                    for file_type, data in dataset.items():
                        f = file_dict[file_type]
                        create_hdf5_dataset_if_missing(f, data, n_games, n_runs, n_checkpoints)
                        write_to_hdf5(f, game_idx, run_idx, ckpt_idx, data)

                    del dataset  # free memory
    finally:
        close_hdf5_files(file_dict)
    
    # Rename temp files to final files
    # If anything crashes before this, this won't happen, so we won't get partial files.
    # Otherwise, when we check if these files exist, they will, we won't try to create them again, but they will be incomplete.
    for ftype in file_paths:
        tmp_path = tmp_paths[ftype]
        final_path = file_paths[ftype]
        if tmp_path.exists():
            tmp_path.rename(final_path)

    return file_paths


# ---------------- Helper functions to build_hdf5_dataset ---------------- #

def open_hdf5_files(file_paths, obs_exist):
    """
    Open HDF5 files for writing and create parent directories.
    Skip observation if it already exists.
    """
    file_dict = {}
    for file_type, file_path in file_paths.items():
        if file_type == 'observation' and obs_exist:
            continue
        os.makedirs(file_path.parent, exist_ok=True)
        file_dict[file_type] = h5py.File(file_path, 'w')
    return file_dict


def close_hdf5_files(file_dict):
    """Close all opened HDF5 files."""
    for f in file_dict.values():
        f.close()


def create_hdf5_dataset_if_missing(h5_file, data, n_games, n_runs, n_checkpoints):
    """Create dataset inside HDF5 file if it does not exist yet."""
    if 'data' not in h5_file:
        # fallback: create dataset shape dynamically
        shape = (n_games, n_runs, n_checkpoints, *data.shape)
        h5_file.create_dataset('data', shape, dtype=data.dtype)


def write_to_hdf5(h5_file, game_idx, run_idx, ckpt_idx, data):
    """Write a numpy array into the HDF5 dataset at the proper indices."""
    h5_file['data'][game_idx, run_idx, ckpt_idx, ...] = deepcopy(data)


def load_checkpoint_dataset(cfg_dict, game, run, ckpt, obs_exist):
    """
    Loads a single dataset from the specified game, run, and checkpoint.

    :param cfg_dict: Configuration dictionary
    :param game: Game name
    :param run: Run identifier
    :param ckpt: Checkpoint identifier
    :param obs_exist: Whether the observation file already exists (to skip loading it)
    :return: Dictionary of loaded datasets
    """
    print("=" * 50)
    print(game)
    print("=" * 50)
    dataset = {}
    _game = snake_to_camel(game)

    print(f"Loading from game {game}, run {run}, checkpoint {ckpt}")

    for file_type in FILE_TYPES:
        gz_filepath = Path(f"{cfg_dict['data_dir']}/{cfg_dict['dataset_type']}/{_game}/{file_type}_{run}_{ckpt}.npy.gz")

        if file_type == "observation":

            if obs_exist:
                print("Observation has already been prepocessed. Skipping load.")
                continue

            npy_filepath = gz_filepath.with_suffix('.npy')
            if not npy_filepath.exists():
                print(f"Observation file {npy_filepath} does not exist as .npy. Creating from .gz")
                _data = load_from_gz(gz_filepath, cfg_dict)
                np.save(npy_filepath, _data)
                print(f"Saved {npy_filepath} from .gz on disk.")
                del _data  # free memory
            
            data = np.load(npy_filepath)

            if len(data.shape) == 3:  # add channel dimension if missing
                data = np.expand_dims(data, axis=1)  # (N, C, H, W)
            
        elif file_type == "action":
            # RL Unplugged actions are already taken from a minimal action set, so no mapping needed
            data = load_from_gz(gz_filepath, cfg_dict)

        elif file_type == "reward":
            data = load_from_gz(gz_filepath, cfg_dict)

        elif file_type == "terminal":
            data = load_from_gz(gz_filepath, cfg_dict).astype(bool)
            # propagate terminal signals by n_step
            for _ in range(cfg_dict['n_step'] -1):
                data |= np.pad(data[1:], (0, 1))  # shift left by 1 and OR            

        elif file_type == "game_id":
            game_list = sorted(GAMES)  # ensure consistent ordering
            if game in game_list:
                game_id = game_list.index(game)
            else:
                game_id = 0  # default to 0 for any other game
                print(f"Warning: game {game} not found in game list. Defaulting game_id to 0.")
            print("Game ID: ", game_id)
            data = np.full((cfg_dict['samples_per_checkpoint'],), game_id, dtype=np.int32)

        elif file_type == "rtg":
            rewards = dataset['reward']
            dones = dataset['terminal']
            gamma = cfg_dict['gamma']
            n_step = cfg_dict['probe_n_step']
            rtgs = np.zeros_like(rewards)
            for step in reversed(range(n_step)):
                n_step_reward = np.concatenate((rewards[step:], np.zeros(step)))
                n_step_done = np.concatenate((dones[step:], np.zeros(step)))                            
                rtgs = n_step_reward + gamma * rtgs * (1 - n_step_done)
            
            data = rtgs
    
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
        dataset[file_type] = data
            
    return dataset


def snake_to_camel(snake_str):
    """Convert snake_case string to CamelCase."""
    components = snake_str.split('_')
    return ''.join(x.capitalize() for x in components)

def load_from_gz(gz_filepath, cfg_dict):
    """Load a numpy array from a .npy.gz file."""
    g = gzip.GzipFile(gz_filepath)
    _data = np.load(g)
    data = np.copy(_data[:cfg_dict['samples_per_checkpoint']])
    print(f"Using {data.size * data.itemsize / (1024**2):.2f} MB of memory for {gz_filepath.name}")
    del _data  # free memory
    return data


# ------------------------------------------------------------------------- #
# Pretrain game list, used for getting indices. DON'T CHANGE.
# PRETRAIN_GAMES = ['amidar', 'atlantis', 'bank_heist', 'battle_zone', 'boxing', 
#         'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber',
#         'demon_attack', 'double_dunk', 'enduro', 'fishing_derby', 'freeway', 
#         'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey',
#         'jamesbond', 'kangaroo', 'krull', 'kung_fu_master', 'ms_pacman', 
#         'name_this_game', 'phoenix', 'qbert', 'road_runner', 'robotank',
#         'space_invaders', 'star_gunner', 'time_pilot', 'up_n_down', 'video_pinball',
#         'wizard_of_wor', 'yars_revenge', 'zaxxon']

GAMES = ['amidar', 'atlantis', 'bank_heist', 'battle_zone', 'boxing', 
        'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber',
        'demon_attack', 'double_dunk', 'enduro', 'fishing_derby', 'freeway', 
        'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey',
        'jamesbond', 'kangaroo', 'krull', 'kung_fu_master', 'ms_pacman', 
        'name_this_game', 'phoenix', 'qbert', 'road_runner', 'robotank',
        'space_invaders', 'star_gunner', 'time_pilot', 'up_n_down', 'video_pinball',
        'wizard_of_wor', 'yars_revenge', 'zaxxon', # 38 pretrain games

        'alien', 'assault', 'asterix', 'beam_rider', 
        'pong', 'pooyan', 'riverraid', 'seaquest', # 8 near-ood games

        'human_cannonball','basic_math','klax','othello','surround' # 5 far-ood games
]

# OLD
# PRETRAIN_GAME_LIST = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', \
#                       'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', \
#                       'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', \
#                       'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk', \
#                       'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', \
#                       'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond', \
#                       'journey_escape', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', \
#                       'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', \
#                       'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', \
#                       'robotank', 'seaquest', 'skiing', 'solaris', 'space_invaders', \
#                       'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', \
#                       'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']


# ------------------------------------------------------------------------- #
def get_dataset_class(cfg):
    """
    Get the dataset class based on the dataset type in the configuration.

    :param cfg: Configuration dictionary
    :return: Dataset class
    """
    dataset_class = cfg['dataset_class']
    
    if dataset_class == 'default':
        return DefaultDataset
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class}")


def get_dataloader(cfg, dataset):
    """
    Create a DataLoader and Sampler based on the configuration and dataset.

    :param cfg: Configuration dictionary
    :param dataset: Dataset object
    :return: Tuple of (DataLoader, Sampler)
    """
    prefetch_factor = cfg['prefetch_factor']
    if cfg['num_workers'] == 0:
        prefetch_factor = None
    
    if cfg['distributed']:
        sampler = DistributedSampler(dataset, shuffle=cfg['shuffle'])
        shuffle = False
    else:
        sampler = None
        shuffle = cfg['shuffle']

    dataloader = DataLoader(dataset,
                            batch_size=cfg['batch_size'],
                            num_workers=cfg['num_workers'],
                            pin_memory=cfg['pin_memory'],
                            shuffle=shuffle,
                            sampler=sampler,
                            drop_last=False,
                            prefetch_factor=prefetch_factor)
    
    return dataloader, sampler