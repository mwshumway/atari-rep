import torch
import numpy as np
import random

def set_global_seeds(seed):
    # torch.backends.cudnn.enabled = False # https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860#h-2-cudagraphs-cause-oom-9
    torch.backends.cudnn.deterministic = True
    # https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16
    torch.backends.cudnn.benchmark = True 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)