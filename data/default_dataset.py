"""
src/data/default_dataset.py
============================
This file defines the DefaultDataset class, which is a subclass of BaseDataset.
"""
from .base_dataset import BaseDataset


class DefaultDataset(BaseDataset):
    name = "default"
    def __init__(self, file_paths, cfg_dict):
        super().__init__(file_paths, cfg_dict)
    
    def __getitem__(self, idx):
        game_idx, run_idx, ckpt_idx, idx = self.get_indexes(idx)  # unravel, see BaseDataset

        start_idx = idx
        end_idx = start_idx + self.t_step + (self.frame - 1)

        # Create slices for current and next observations
        slc = slice(start_idx, end_idx)
        nslc = slice(start_idx + self.n_step, end_idx + self.n_step)  # next observation slice is offset by n_step

        slice_dict = {file_type: slc for file_type in self.file_paths.keys()}
        slice_dict['next_observation'] = nslc

        item_dict = {}
        for file_type, _slice in slice_dict.items():
            item_dict[file_type] = self.access_file(file_type, game_idx, run_idx, ckpt_idx, _slice)

        return item_dict
