import math

from torch.optim import Optimizer

from .lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """Initializes a WarmupCosineLR learning rate scheduler.

        Args:
            optimizer (`Optimizer`): The optimizer for which to adjust the learning rate.
            warmup_iters (`int`): Number of warm-up iterations.
            lr_decay_iters (`int`): Number of iterations for learning rate decay (approximately equal to MAX_ITERS).
            min_lr (`float`): Minimum learning rate.
            last_epoch (`int`, optional): The index of the last epoch. Default is -1.
            verbose (`bool`, optional): If True, prints a message when the learning rate is updated. Default is False.
        """
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        # refer to https://github.com/karpathy/nanoGPT/blob/7fe4a099ad2a4654f96a51c0736ecf347149c34c/train.py#L226
        initial_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]

        it = self.last_epoch

        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return [lr * it / self.warmup_iters for lr in initial_lrs]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return [self.min_lr for _ in initial_lrs]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

        return [self.min_lr + coeff * (lr - self.min_lr) for lr in initial_lrs]
