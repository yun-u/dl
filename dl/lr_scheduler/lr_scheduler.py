try:
    # try to import LRScheduler from torch.optim.lr_scheduler (for PyTorch 2.0+)
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # if ImportError occurs, import _LRScheduler (for PyTorch 1.0-1.13)
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa: F401
