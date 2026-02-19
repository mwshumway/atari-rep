"""
src/probing/datasets/base.py
=============================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from abc import ABCMeta, abstractmethod
import tqdm
from einops import rearrange
import os


class BaseProbeDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for probe datasets that extract represtations from a model.

    :param model: the model to extract representations from
    :param data_loader: the data loader providing input data
    :param device: the device to run the model on
    :param feature_extractor: string specifying which part of the model to extract features from
    :param cfg: configuration object with necessary parameters
    """

    def __init__(self, 
                 model,
                 dataloader,
                 device,
                 feature_extractor,
                 cfg,
                 extract=True):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.save_rep = cfg.probing.save_representation

        self.representations = torch.tensor([])
        self.targets = torch.tensor([])
        self.game_ids = torch.tensor([])

        # get representations once during initialization
        if extract:
            self._extract_representations()
    
    def _extract_representations(self):
        """
        Extract representations from the neck of the model for all data.
        """
        # Check if we've saved representations before
        if os.path.exists(self.cfg.probing.saved_representation_path):
            self._load_representation_data(self.cfg.probing.saved_representation_path)
            return
        
        print("Extracting representations...")
        representations, targets, game_ids = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(self.dataloader, desc="Representation Extraction"):
                for key, value in batch.items():
                    batch[key] = value.to(self.device)

                batch = self._collate(batch)
                obs = batch['obs']
                game_id = batch['game_id']

                if self.feature_extractor == 'backbone':
                    feat, _ = self.model.backbone(obs)
                elif self.feature_extractor == 'neck':
                    feat, _ = self.model.backbone(obs)
                    feat, _ = self.model.neck(feat, game_id)
                    feat = feat[:, 0, :]  # take representation corresponding to the first frame
                    feat = feat.unsqueeze(1)
                else:
                    raise ValueError(f"Unknown feature extractor: {self.feature_extractor}")
                
                # free memory
                del batch['obs']
                del batch['next_obs']

                # Flatten temporal dimension if present
                # (batch, time, ...) -> (batch*time, 1, ...)
                feat = feat.flatten(start_dim=0, end_dim=1).unsqueeze(1)

                # Extract target labels
                target = self._extract_target(batch)
                target = target.flatten(start_dim=0, end_dim=1).unsqueeze(1)

                # Game IDs
                game_id = game_id.flatten(start_dim=0, end_dim=1).unsqueeze(1)

                representations.append(feat)
                targets.append(target)
                game_ids.append(game_id)

        self.representations = torch.cat(representations, dim=0)
        self.targets = torch.cat(targets, dim=0)
        self.game_ids = torch.cat(game_ids, dim=0)

        print(f"Extracted {len(self.representations)} samples")
        print(f"Representation shape: {self.representations.shape}")
        print(f"Target shape: {self.targets.shape}")

        # Save representations for future use
        if self.save_rep: self._save_representation_data()

    @abstractmethod
    def _extract_target(self, batch) -> torch.Tensor:
        """Extract the target labels from the batch. To be implemented in subclasses."""
        pass

    @property
    @abstractmethod
    def probe_type(self) -> str:
        """return the type of probing task (e.g., 'reward', 'action', etc.)"""
        pass

    def __len__(self) -> int:
        return len(self.representations)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'representation': self.representations[idx],
            'target': self.targets[idx],
            'game_id': self.game_ids[idx]
        }

    def _collate(self, batch):
        """
        [params] 
            observation: (n, t+f-1, c, h, w) 
            next_observation: (n, t+f-1, c, h, w)
            action:   (n, t+f-1)
            reward:   (n, t+f-1)
            terminal: (n, n+t+f-1) * different n's (batch vs n_step)
            game_id:  (n, t+f-1)            
        [returns] 
            (c = 1 in ATARI)
            obs:      (n, t, f, c, h, w) 
            next_obs: (n, t, f, c, h, w)
            action:   (n, t)
            reward:   (n, t)
            done:     (n, n+t)
            game_id:  (n, t)    
        """
        f = self.cfg.dataloader.frame
        obs = batch['observation']
        action = batch['action']
        reward = batch['reward']
        rtg = batch['rtg']
        done = batch['terminal']
        game_id = batch['game_id']
        next_obs = batch['next_observation']

        # process data-format
        obs = rearrange(obs, 'n tf c h w -> n tf 1 c h w')
        obs = obs.repeat(1, 1, f, 1, 1, 1)
        next_obs = rearrange(next_obs, 'n tf c h w -> n tf 1 c h w')
        next_obs = next_obs.repeat(1, 1, f, 1, 1, 1)
        action = action.long()
        reward = torch.nan_to_num(reward).sign()
        rtg = rtg.float()
        done = done.bool()
        game_id = game_id.long()

        # frame-stack
        if f != 1:
            for i in range(1, f):
                obs[:, :, i] = obs[:, :, i].roll(-i, 1)
                next_obs[:, :, i] = next_obs[:, :, i].roll(-i, 1)
            obs = obs[:, :-(f-1)]
            next_obs = next_obs[:, :-(f-1)]
            action = action[:, f-1:]
            rtg = rtg[:, f-1:]
            reward = reward[:, f-1:]
            done = done[:, f-1:]
            game_id = game_id[:, f-1:]
            
        # lazy frame to float
        obs = obs / 255.0
        next_obs = next_obs / 255.0
            
        batch = {
            'obs': obs,
            'next_obs': next_obs,
            'act': action,
            'rew': reward,
            'rtg': rtg,
            'done': done,
            'game_id': game_id,                            
        }            
            
        return batch

    def _load_representation_data(self, path: str):
        "Load saved representation data from disk."
        print(f"Loading saved representations from {self.cfg.probing.saved_representation_path}")
        data = torch.load(self.cfg.probing.saved_representation_path)
        self.representations = data['representations']
        self.targets = data['targets']
        self.game_ids = data['game_ids']
        print(f"Loaded {len(self.representations)} samples")
        print(f"Representation shape: {self.representations.shape}")
        print(f"Target shape: {self.targets.shape}")
    
    def _save_representation_data(self):
        "Save representation data to disk."
        os.makedirs(os.path.dirname(self.cfg.probing.saved_representation_path), exist_ok=True)
        torch.save({
            'representations': self.representations,
            'targets': self.targets,
            'game_ids': self.game_ids
        }, self.cfg.probing.saved_representation_path)
        print(f"Saved representations to {self.cfg.probing.saved_representation_path}")

