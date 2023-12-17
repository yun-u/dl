import torch

from dl.lr_scheduler import WarmupCosineLR


def test_warmup_cosine_lr():
    warmup_iters = 5
    max_iters = 10

    # scratch
    lrs = []
    opt = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=3e-4)
    lr_scheduler = WarmupCosineLR(
        opt, warmup_iters=warmup_iters, lr_decay_iters=max_iters, min_lr=1e-5
    )
    for _ in range(max_iters):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        lr_scheduler.step()

    # resume
    new_lrs = []
    opt = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=3e-4)
    lr_scheduler = WarmupCosineLR(
        opt, warmup_iters=warmup_iters, lr_decay_iters=max_iters, min_lr=1e-5
    )
    for _ in range(max_iters // 2):
        new_lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        lr_scheduler.step()

    lr_scheduler = WarmupCosineLR(
        opt,
        warmup_iters=warmup_iters,
        lr_decay_iters=max_iters,
        min_lr=1e-5,
        last_epoch=max_iters // 2 - 1,
    )
    for _ in range(max_iters // 2 + max_iters % 2):
        new_lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        lr_scheduler.step()

    assert lrs == new_lrs
