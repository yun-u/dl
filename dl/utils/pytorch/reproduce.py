import random

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "worker_init_function",
]


def seed_everything(seed: int):
    # refer to https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_function(worker_id: int):
    initial_seed = torch.initial_seed() % 2**32
    seed_everything(initial_seed + worker_id)
