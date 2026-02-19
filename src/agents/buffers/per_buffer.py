import torch
import torch.nn as nn
import numpy as np
from collections import deque
from .base import BaseBuffer
from einops import rearrange
import time
import os
import gzip
import pickle


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.max = 1  # Initial max value to return (1 = 1^ω), default transition priority is set to max

     # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

     # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, value):
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    def total(self):
        return self.sum_tree[0]
    

class PERBuffer(BaseBuffer):
    name = 'per_buffer'
    def __init__(self, obs_shape, action_size, size, prior_exp, max_n_step, device, save_backbone_feat, backbone_feat_shape):
        super().__init__()
        self.size = size
        self.prior_exp = prior_exp
        self.max_n_step = max_n_step
        self.save_backbone_feat = save_backbone_feat
        self.backbone_feat_shape = backbone_feat_shape
        self.device = device

        self.n_step_transitions = deque(maxlen=max_n_step)
        if self.save_backbone_feat:
            feat_shape = (size, *backbone_feat_shape)
            self.feat_buffer = np.zeros(feat_shape, dtype=np.float32)
        else:
            self.obs_buffer = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.act_buffer = np.zeros(size, dtype=np.int64)
        self.rew_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.float32)
        self.segment_tree = SegmentTree(size)
        self.num_in_buffer = 0
        self.buffer_idx = 0
        self.ckpt = 1

    def store(self, feat, action, reward, done):
        """
        feat may be either raw observation or backbone feature
        """

        self.n_step_transitions.append((feat, action, reward, done))
        if len(self.n_step_transitions) < self.max_n_step:
            return
        feat, action, reward, done = self.n_step_transitions[0]

        b_idx = self.buffer_idx
        if self.save_backbone_feat:
            self.feat_buffer[b_idx] = feat
        else:
            self.obs_buffer[b_idx] = feat
        self.act_buffer[b_idx] = action
        self.rew_buffer[b_idx] = reward
        self.done_buffer[b_idx] = done

        # store new transition with maximum priority
        self.segment_tree.append(value=self.segment_tree.max)

        # increase buffer count
        self.num_in_buffer = min(self.num_in_buffer+1, self.size)
        b_idx += 1
        if b_idx == self.size:
            b_idx = 0
        self.buffer_idx = b_idx

    # Returns a valid sample from each segment
    def _get_idxs_from_segments(self, batch_size):
        p_total = self.segment_tree.total() # sum of the priorities
        segment_length = p_total / batch_size # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False

        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts # Uniformly sample from within all segments
            probs, data_idxs, tree_idxs = self.segment_tree.find(samples) # Retrieve samples from tree with un-normalised probability

            # extra conservative around buffer index 0
            # n_step must be stacked before sampling
            if np.all(probs != 0) and np.all(data_idxs < (self.size - self.max_n_step)):
                valid = True
        
        return data_idxs, tree_idxs, probs

    def sample(self, batch_size, n_step=10, gamma=0.99, prior_weight=1.0):
        assert self.num_in_buffer >= batch_size, \
            f'Replay buffer does not have enough transitions to sample. ' \
            f'num_in_buffer: {self.num_in_buffer}, batch_size: {batch_size}'
        data_idxs, tree_idxs, probs = self._get_idxs_from_segments(batch_size)

        # vectorize n-step returns computation
        n_step_idxs = data_idxs[:, np.newaxis] + np.arange(n_step)

        # Fetch all rewards and dones at once
        n_step_rew_batch = self.rew_buffer[n_step_idxs] # (batch_size, n_step)
        n_step_done_batch = self.done_buffer[n_step_idxs] # (batch_size, n_step)

        gamma_powers = gamma ** np.arange(n_step) # [1, gamma, gamma^2, ..., gamma^(n_step-1)]

        not_done = 1.0 - n_step_done_batch
        done_mask = np.ones_like(n_step_done_batch)
        for i in range(1, n_step):
            done_mask[:, i] = done_mask[:, i-1] * not_done[:, i-1]

        # Compute discounted returns G = sum(gamma^k * r_k * done_mask_k)
        discounted_rewards = n_step_rew_batch * gamma_powers * done_mask
        G = np.sum(discounted_rewards, axis=1)

        # Find first done or use last step
        # done_positions: index of first done, or n_step if no done)
        done_positions = np.argmax(n_step_done_batch, axis=1)
        has_done = np.any(n_step_done_batch > 0, axis=1)
        done_positions = np.where(has_done, done_positions, n_step - 1)

        # terminal states is wen any done occured
        done = n_step_done_batch[np.arange(batch_size), done_positions]

        # offset for next feat
        next_feat_offset = done_positions + 1

        # single reward for TD target (reward at done position or last step)
        rew = n_step_rew_batch[np.arange(batch_size), done_positions]

        # Get transitions
        feat_batch = self.feat_buffer[data_idxs] if self.save_backbone_feat else self.obs_buffer[data_idxs]
        act_batch = self.act_buffer[data_idxs]
        rew_batch = rew
        done_batch = done
        G_batch = G
        next_feat_batch = self.feat_buffer[data_idxs + next_feat_offset] if self.save_backbone_feat else self.obs_buffer[data_idxs + next_feat_offset]

        if self.save_backbone_feat:
            feat_batch = torch.from_numpy(feat_batch).to(self.device, non_blocking=True).unsqueeze(1)
            next_feat_batch = torch.from_numpy(next_feat_batch).to(self.device, non_blocking=True).unsqueeze(1)
        else:
            feat_batch = torch.from_numpy(feat_batch.astype(np.float32)).to(self.device, non_blocking=True).unsqueeze(1)
            next_feat_batch = torch.from_numpy(next_feat_batch.astype(np.float32)).to(self.device, non_blocking=True).unsqueeze(1)
        act_batch = torch.from_numpy(act_batch).to(self.device, dtype=torch.long, non_blocking=True)
        rew_batch = torch.from_numpy(rew_batch).to(self.device, dtype=torch.float32, non_blocking=True)
        done_batch = torch.from_numpy(done_batch).to(self.device, dtype=torch.float32, non_blocking=True)
        G_batch = torch.from_numpy(G_batch).to(self.device, dtype=torch.float32, non_blocking=True)

        # compute importance weights
        p_total = self.segment_tree.total()
        N = self.num_in_buffer
        probs = probs / p_total
        weights = (1.0 / (probs * N + 1e-5)) ** prior_weight

        # re-normalize by max weight (makes updates scale consistent w.r.t learning rate)
        weights = weights / np.max(weights)
        weights = torch.from_numpy(weights).to(self.device, dtype=torch.float32, non_blocking=True)

        batch = {
            'feat': feat_batch,
            'act': act_batch,
            'rew': rew_batch,
            'done': done_batch,
            'G': G_batch,
            'next_feat': next_feat_batch,
            'n_step': n_step,
            'gamma': gamma,
            'tree_idxs': tree_idxs,
            'weights': weights,
        }
        return batch
    
    def encode_obs(self, obs, prediction=False):
        if isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        else:
            obs = np.array(obs, dtype=np.float32)

        # prediction: batch-size: 1
        if prediction:
            obs = np.expand_dims(obs, 0)

        obs = np.expand_dims(obs, 1)
        obs = torch.FloatTensor(obs).to(self.device, non_blocking=True)

        return obs / 255.0

    def update_priorities(self, idxs, priorities):
        """Optimized priority update"""
        # Clip priorities to avoid numerical issues
        priorities = np.clip(priorities, 1e-6, None)
        priorities = np.power(priorities, self.prior_exp)
        self.segment_tree.update(idxs, priorities)
        
    def save_buffer(self, buffer_dir, game, run_id, ckpt=None):
        dir = f"{buffer_dir}/{game}/run{run_id}/"
        ckpt = self.ckpt if ckpt is None else ckpt
        postfix = f"_{ckpt}.gz"

        os.makedirs(dir, exist_ok=True)

        # --- Save arrays ---
        if self.save_backbone_feat:
            with gzip.open(dir + 'backbone_feat' + postfix, 'wb') as f:
                np.save(f, self.feat_buffer)
        else:
            with gzip.open(dir + 'observation' + postfix, 'wb') as f:
                np.save(f, self.obs_buffer)

        with gzip.open(dir + 'action' + postfix, 'wb') as f:
            np.save(f, self.act_buffer)

        with gzip.open(dir + 'reward' + postfix, 'wb') as f:
            np.save(f, self.rew_buffer)

        with gzip.open(dir + 'terminal' + postfix, 'wb') as f:
            np.save(f, self.done_buffer)

        # --- Save segment tree ---
        with gzip.open(dir + 'segment_tree' + postfix.replace('.gz','.pkl'), 'wb') as f:
            pickle.dump(self.segment_tree, f, protocol=pickle.HIGHEST_PROTOCOL)

        # --- Save metadata ---
        metadata = {
            'num_in_buffer': self.num_in_buffer,
            'buffer_idx': self.buffer_idx,
            'ckpt': self.ckpt,
            'prior_exp': self.prior_exp,
            'max_n_step': self.max_n_step,
            'size': self.size,
            'n_step_transitions': list(self.n_step_transitions),
        }

        with gzip.open(dir + 'metadata' + postfix.replace('.gz','.pkl'), 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved buffer checkpoint {ckpt} to {dir}")


    def load_buffer(self, buffer_dir, game, run_id, ckpt='latest'):
        dir = f"{buffer_dir}/{game}/run{run_id}/"
        postfix = f"_{ckpt}.gz"

        # --- Load arrays ---
        if self.save_backbone_feat:
            with gzip.open(dir + 'backbone_feat' + postfix, 'rb') as f:
                self.feat_buffer = np.load(f)
        else:
            with gzip.open(dir + 'observation' + postfix, 'rb') as f:
                self.obs_buffer = np.load(f)

        with gzip.open(dir + 'action' + postfix, 'rb') as f:
            self.act_buffer = np.load(f)

        with gzip.open(dir + 'reward' + postfix, 'rb') as f:
            self.rew_buffer = np.load(f)

        with gzip.open(dir + 'terminal' + postfix, 'rb') as f:
            self.done_buffer = np.load(f)

        # --- Load segment tree ---
        with gzip.open(dir + 'segment_tree' + postfix.replace('.gz','.pkl'), 'rb') as f:
            self.segment_tree = pickle.load(f)

        # --- Load metadata ---
        with gzip.open(dir + 'metadata' + postfix.replace('.gz','.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        self.num_in_buffer = metadata['num_in_buffer']
        self.buffer_idx = metadata['buffer_idx']
        self.ckpt = metadata['ckpt']
        self.prior_exp = metadata['prior_exp']
        self.max_n_step = metadata['max_n_step']
        self.size = metadata['size']
        
        # important: restore n-step sequence
        self.n_step_transitions = deque(metadata['n_step_transitions'],
                                        maxlen=self.max_n_step)

        print(f"Loaded buffer checkpoint {ckpt} from {dir}")
