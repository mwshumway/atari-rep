from copy import deepcopy
import os

from src.data.data_utils import data_downloaded, build_hdf5_dataset, get_dataset_class, get_dataloader
from src.data.convert_tf_files import convert_tf_files

def download_data(args):
    if not data_downloaded(args):
        print(f"{args.data.dataset_name} dataset not found. Starting download and conversion.")
        convert_tf_files(args)
    else:
        print(f"{args.data.dataset_name} dataset already exists. Skipping download.")


def build_dataloader(args):
    eval_ratio = args.data.eval_ratio

    eval_args = deepcopy(args)
    eval_args.data.samples_per_checkpoint = int(eval_ratio * args.data.samples_per_checkpoint)
    eval_args.data.shuffle = False
    eval_args.data.distributed = False

    # Process/prepare dataset and get file paths
    # instead of having all the games split up into different folders, we just have one folder
    # of organization: subdir_name/oberservation_{samples_per_checkpoint}.hdf5
    # or subdir_name/{other_file_type}_{samples_per_checkpoint}_{n_step}_{gamma}.hdf5
    file_paths = build_hdf5_dataset(args)  # builds if not already built and returns file paths
    print("HDF5 dataset is ready.")

    # Instantiate dataset class based on configuration
    dataset_class = get_dataset_class(args)
    train_dataset = dataset_class(file_paths=file_paths, args=args)
    eval_dataset = dataset_class(file_paths=file_paths, args=eval_args)

    print(len(train_dataset), len(eval_dataset))

    train_dataloader, train_sampler = get_dataloader(args, train_dataset)
    eval_dataloader, eval_sampler = get_dataloader(eval_args, eval_dataset)

    return train_dataloader, train_sampler, eval_dataloader, eval_sampler
