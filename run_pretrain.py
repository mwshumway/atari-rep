import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch\\.cuda\\.amp\\.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*Unsupported unwinding pattern.*")
warnings.filterwarnings("ignore", message="Profiler function <class 'torch.autograd.profiler.record_function'> will be ignored")


import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._logging").setLevel(logging.ERROR)

import wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
import tyro
import os

from src.utils.seed import set_global_seeds
from src.data import build_dataloader, download_data
from src.logger import PretrainLogger
from src.model import build_model
from src.pretrain import build_trainer
from configs import BaseConfig

def run(cfg):
    # Set random seed
    set_global_seeds(cfg.seed)

    download_data(cfg)
    os.environ['MASTER_ADDR'] = cfg.master_addr 
    os.environ['MASTER_PORT'] = cfg.master_port

    if cfg.num_gpus_per_node > 1:
        cfg.data.distributed = True
        cfg.pretrain.distributed = True
        wandb.setup()
        mp.spawn(run_worker, # type: ignore
                 nprocs=cfg.num_gpus_per_node, 
                 args=(cfg.num_gpus_per_node, cfg))

    else:
        cfg.data.distributed = False
        cfg.pretrain.distributed = False
        run_worker(gpu_id=0, num_gpus_per_node=1, cfg=cfg)


def run_worker(gpu_id, num_gpus_per_node, cfg):
    cfg.device = 'cuda:' + str(gpu_id)
    torch.cuda.set_device(gpu_id)
    cfg.rank = cfg.rank * num_gpus_per_node + gpu_id
    device = torch.device(cfg.device)

    print("Rank {}, Use {} for training".format(cfg.rank, cfg.device))

    if num_gpus_per_node > 1:
        print('Train with distributed data parallel')
        cfg.data.batch_size = cfg.data.batch_size // num_gpus_per_node
        dist.init_process_group(backend='nccl', 
                                init_method='env://',
                                world_size=cfg.num_gpus_per_node, 
                                rank=cfg.rank,
                                timeout=datetime.timedelta(minutes=90)) # evaluation takes longer with probes now
    else:
        print('Train without distributed data parallel')
    
    torch.set_num_threads(1)
    train_loader, train_sampler, eval_loader, eval_sampler = build_dataloader(cfg)

    print(f"number of training batches: {len(train_loader)}")
    print(f"number of eval batches: {len(eval_loader)}")

    logger = PretrainLogger(cfg)

    model = build_model(cfg, device)

    trainer = build_trainer(cfg, device, train_loader, train_sampler, eval_loader, eval_sampler, logger, model)

    trainer.train()

    if cfg.rank == 0 and cfg.wandb.enabled:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    cfg = tyro.cli(BaseConfig)
    run(cfg)