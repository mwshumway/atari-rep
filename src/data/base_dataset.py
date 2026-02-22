"""
src/data/base_dataset.py
========================

This file defines the BaseDataset class, which serves as a foundational class for creating custom datasets.

The main idea here is that instead of loading all data into memory at once, we load data on-the-fly when __getitem__ is called.
"""

from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
import os
import h5py
import torch


class BaseDataset(Dataset):
    name = 'base'

    def __init__(self, file_paths, args):
        self.file_paths = file_paths
        # attributes we need from the args object
        self.games = args.games
        self.runs = args.data.runs
        self.checkpoints = args.data.checkpoints
        self.samples_per_checkpoint = args.data.samples_per_checkpoint
        self.frame = args.frame
        self.t_step = args.t_step
        self.n_step = args.n_step
        self.gamma = args.gamma

        # derived attributes
        self.n_games = len(self.games)
        self.n_runs = len(self.runs)
        self.n_ckpts = len(self.checkpoints)
        self.file_paths['next_observation'] = self.file_paths['observation']
        _, self.file_suffix = os.path.splitext(self.file_paths['observation'])

        # we compute the effective size of each checkpoint
        # it's the number of samples we can actually use given the frame, t_step, and n_step
        if self.samples_per_checkpoint > 0:
            self.effective_size = self.samples_per_checkpoint - (self.frame - 1) - (self.t_step - 1) - self.n_step
        else:
            self.effective_size = 0  # handle case where samples_per_checkpoint is 0 or negative

    @classmethod
    def get_name(cls):
        return cls.name
    
    def __len__(self):
        return self.n_games * self.n_runs * self.n_ckpts * self.effective_size
    
    def get_indexes(self, index):
        """
        Given a global index, return the corresponding (game_idx, run_idx, ckpt_idx, local_index)
        where local_index is the index within the effective size of the checkpoint.

        It's like unraveling a multi-dimensional index, see __len__ and attributes above.
        """
        game_idx = index // (self.n_runs * self.n_ckpts * self.effective_size)
        index %= self.n_runs * self.n_ckpts * self.effective_size

        run_idx = index // (self.n_ckpts * self.effective_size)
        index %= self.n_ckpts * self.effective_size

        ckpt_idx = index // self.effective_size
        index %= self.effective_size

        return (game_idx, run_idx, ckpt_idx, index)

    def load_hdf5(self):
        """
        Load the HDF5 files into memory. Called only once, in access_file().
        """
        self.dataset_dict = {}
        for file_type, file_path in self.file_paths.items():
            f = h5py.File(file_path, 'r')
            self.dataset_dict[file_type] = f['data']  # 'data' is the name of the dataset within the HDF5 file 
    
    @abstractmethod
    def access_file(self, file_type, game_idx, run_idx, ckpt_idx, slice_obj):
        """
        Abstract method to access the data file.

        Assumes file extension is .npy
        """
        if self.file_suffix == '.npy':
            data = np.load(self.file_paths[file_type], mmap_mode='r')
        
        elif self.file_suffix == '.hdf5':
            if not hasattr(self, 'dataset_dict'):
                self.load_hdf5()  # load HDF5 files into memory if not already loaded
            data = self.dataset_dict[file_type]
        else:
            raise NotImplementedError(f"File suffix {self.file_suffix} not supported.")

        item = data[game_idx, run_idx, ckpt_idx, slice_obj]
        item = torch.tensor(item)
        del data  # free memory

        return item

    def __getitem__(self, index):
        pass  # To be implemented in subclasses