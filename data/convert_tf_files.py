"""
src/data/convert_tf_files.py
============================

Script to convert TFRecord files to the expected format.

Expected format:

{data_dir}/
    {game}/
        observation_{run}_{checkpoint}.npy.gz
        action_{run}_{checkpoint}.npy.gz
        reward_{run}_{checkpoint}.npy.gz
        terminal_{run}_{checkpoint}.npy.gz

A. observation_{run}_{checkpoint}.npy.gz:
    Shape: (num_transitions, 1, 84, 84)
    Dtype: uint8
    Description: Grayscale Atari frames, resized to 84x84.
B. action_{run}_{checkpoint}.npy.gz
    Shape: (num_transitions,)
    Dtype: uint8
    Description: Actions taken by the agent.
C. reward_{run}_{checkpoint}.npy.gz
    Shape: (num_transitions,)
    Dtype: float32, clipped to [-1, 1]
    Description: Rewards received after taking actions.
D. terminal_{run}_{checkpoint}.npy
    Shape: (num_transitions,)
    Dtype: uint8
    Description: Whether the state is terminal.
        1 or True: episode ended at this step.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import gzip
import shutil

# ----------------------------------------
# Prevent TF from using too many resources
# ----------------------------------------
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# TensorFlow GCS access test (fail fast)
def check_gcs_access():
    """
    Check if TensorFlow can access GCS paths.
    """
    test_path = "gs://rl_unplugged/atari_episodes_ordered/Alien/run_1-00000-of-00050"
    try:
        tf.io.gfile.exists(test_path)
        print("GCS access test passed.")
    except Exception as e:
        raise RuntimeError(f"TensorFlow cannot access GCS paths: {e}")


def get_gcs_path(game, run, shard):
    """
    Get the correct GCS path for RLU Atari data.

    For 'Carnival', 'Gravitar', and 'StarGunner', there are 49 shards (indices 0-48).
    All other games have 50 shards (indices 0-49).

    For the three exceptions, assumes the last shard comes in as 49, and maps it to 48.

    :param game: Name of the Atari game.
    :param run: Run number (1-5).
    :param shard: Shard index (0-49 for most games, 0-48 for exceptions).
    """
    base = "gs://rl_unplugged/atari_episodes_ordered"
    exceptions = ["Carnival", "Gravitar", "StarGunner"]  # Games that say 'of-00049' instead of 'of-00050'
    
    total = 49 if game in exceptions else 50
    if shard == 49 and game in exceptions:
        shard = 48  # Map the last shard to 48 for these games

    shard_str = f"{shard:05d}-of-{total:05d}"
    gcs_path = f"{base}/{game}/run_{run}-{shard_str}"

    if not tf.io.gfile.exists(gcs_path):
        # Fallback to the non-ordered path if the ordered one is missing
        base = "gs://rl_unplugged/atari_episodes"
        gcs_path = f"{base}/{game}/run_{run}-{shard_str}"

    if not tf.io.gfile.exists(gcs_path):
        raise FileNotFoundError(f"GCS path not found for {game}, run {run}, shard {shard}: {gcs_path}")

    return gcs_path


def create_stream_dataset(game, run, shard):
    """
    Create a TFRecord dataset streaming from GCS.
    
    :param game: str, name of the Atari game
    :param run: int, run number (1-5)
    :param shard: int, shard number (0-99)
    """
    gcs_path = get_gcs_path(game, run, shard)
    return tf.data.TFRecordDataset(gcs_path, compression_type="GZIP", num_parallel_reads=1)  # num_parallel_reads=1 to avoid too many threads

def parse_tfrecord(raw_record):
    """
    Parse RLU tfrecord.
    
    :param raw_record: raw tfrecord
    """
    feature_description = {
        "observations": tf.io.VarLenFeature(tf.string),
        "actions": tf.io.VarLenFeature(tf.int64),
        "discounts": tf.io.VarLenFeature(tf.float32),
        "clipped_rewards": tf.io.VarLenFeature(tf.float32),
        "episode_idx": tf.io.FixedLenFeature([], tf.int64),
        "checkpoint_idx": tf.io.FixedLenFeature([], tf.int64),
    }

    ex = tf.io.parse_single_example(raw_record, feature_description)

    # Decode sequence fields
    observations = tf.map_fn(
        lambda x: tf.io.decode_image(x, channels=1, dtype=tf.uint8),
        ex["observations"].values,
        fn_output_signature=tf.uint8
    )

    return {
        "observations": observations.numpy(),
        "actions": ex["actions"].values.numpy().astype(np.float64),
        "discounts": ex["discounts"].values.numpy(),
        "rewards": ex["clipped_rewards"].values.numpy().astype(np.float64),
        "episode_idx": ex["episode_idx"].numpy(),
        "checkpoint_idx": ex["checkpoint_idx"].numpy(),
    }

def save_compressed_array(filepath, array):
    """
    Save a numpy array as a gzipped .npy file.
    """
    tmp_path = filepath + ".tmp.npy"
    np.save(tmp_path, array)
    
    with open(tmp_path, "rb") as f_in:
        with gzip.open(filepath + ".npy.gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)  # type: ignore
    
    os.remove(tmp_path)


def build_first_n_transitions(game, run, checkpoints, n_transitions, output_dir):
    """
    Build the first n transitions for a given game and run, across specified checkpoints.

    :param game: Name of the Atari game.
    :param run: Run number (1-5).
    :param checkpoints: List of checkpoint indices to include.
    :param n_transitions: Total number of transitions to collect.
    :param output_dir: Directory to save the output files.
    """
    game_dir = os.path.join(output_dir, game)
    os.makedirs(game_dir, exist_ok=True)

    for ckpt in checkpoints:
        out_files = {
            "observation": os.path.join(game_dir, f"observation_{run}_{ckpt}"),
            "action": os.path.join(game_dir, f"action_{run}_{ckpt}"),
            "reward": os.path.join(game_dir, f"reward_{run}_{ckpt}"),
            "terminal": os.path.join(game_dir, f"terminal_{run}_{ckpt}"),
        }

        ds = create_stream_dataset(game, run, ckpt)

        obs_list, act_list, rew_list, term_list = [], [], [], []
        total = 0
        pbar = tqdm(ds, desc=f"Processing {game} run {run} checkpoint {ckpt}")

        for raw_record in ds:
            record = parse_tfrecord(raw_record)

            # The following can be uncommented to verify that the checkpoint indices match across the whole shard
            # It is commented out to avoid breaking on 3 exception games (Carnival, Gravitar, StarGunner)
            # Where the last shard index is 48 instead of 49. One could add special handling for these games if desired.
        
            # record_chkpt = record["checkpoint_idx"]
            # assert record_chkpt == ckpt, f"Checkpoint mismatch: {record_chkpt} vs {ckpt}"

            # Process episode
            N = len(record["actions"])
            obs = np.transpose(record["observations"], (0, 3, 1, 2))
            term = (record["discounts"] == 0.0).astype(np.float64)

            obs_list.append(obs)
            act_list.append(record["actions"])
            rew_list.append(record["rewards"])
            term_list.append(term)

            total += N
            pbar.update(N)
            if total >= n_transitions:
                break

        pbar.close()

        # Concatenate and truncate
        obs_arr = np.concatenate(obs_list, axis=0)[:n_transitions]
        act_arr = np.concatenate(act_list, axis=0)[:n_transitions]
        rew_arr = np.concatenate(rew_list, axis=0)[:n_transitions]
        term_arr = np.concatenate(term_list, axis=0)[:n_transitions]

        # Save compressed
        save_compressed_array(out_files["observation"], obs_arr)
        save_compressed_array(out_files["action"], act_arr)
        save_compressed_array(out_files["reward"], rew_arr)
        save_compressed_array(out_files["terminal"], term_arr)


def convert_tf_files(cfg):
    """
    Main function to convert TFRecord files to the expected format.

    :param cfg: Configuration object with necessary parameters.
    """
    check_gcs_access()

    output_dir = os.path.join(cfg.data_dir, cfg.dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    print("=== Starting TFRecord conversion ===")

    for _game in cfg.games:
        game = "".join(w.capitalize() for w in _game.split("_"))

        # Skip games already fully processed
        game_dir = os.path.join(output_dir, game)
        if os.path.exists(game_dir):
            existing_files = os.listdir(game_dir)
            expected_files = len(cfg.runs) * len(cfg.checkpoints) * 4  # 4 files per checkpoint
            if len(existing_files) >= expected_files:
                print(f"All files for {game} already exist. Skipping.")
                continue

        for run in cfg.runs:
            print(f"\nProcessing {game}, run {run}")
            build_first_n_transitions(
                game=game,
                run=run,
                checkpoints=cfg.checkpoints,
                n_transitions=cfg.samples_per_checkpoint,
                output_dir=output_dir
            )   

    print("=== TFRecord conversion completed ===")