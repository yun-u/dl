from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dl.visualize.style import Color

from .lr_scheduler import LRScheduler

__all__ = [
    "visualize_lr_scheduler",
    "visualize_lr",
]


def visualize_lr_scheduler(
    lr_scheduler: Callable[[torch.optim.Optimizer], LRScheduler],
    lr: float,
    num_iter: int,
) -> List[float]:
    param = nn.Parameter(torch.zeros([1]))
    optimizer = torch.optim.SGD([param], lr=lr)
    _lr_scheduler = lr_scheduler(optimizer=optimizer)

    learning_rates = []
    for _ in range(num_iter):
        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)
        optimizer.step()
        _lr_scheduler.step(_)

    visualize_lr(learning_rates, "test")  # lr_scheduler.func.__name__)

    return learning_rates


def visualize_lr(learning_rates: List[float], lr_scheduler_type: Optional[str] = None):
    plt.style.use("dark_background")
    fig, axis = plt.subplots(nrows=1, ncols=1)

    axis.plot(
        range(len(learning_rates)),
        learning_rates,
        color=Color.Dark.BLUE,
        linewidth=0.8,
        marker="o",
        markersize=4.0,
    )

    lr_min, lr_argmin = np.amin(learning_rates), np.argmin(learning_rates)
    lr_max, lr_argmax = np.amax(learning_rates), np.argmax(learning_rates)

    first_idx, last_idx = 0, len(learning_rates) - 1

    coords = set(
        [
            (lr_argmin, lr_min),
            (lr_argmax, lr_max),
            (first_idx, learning_rates[first_idx]),
            (last_idx, learning_rates[last_idx]),
        ]
    )

    for x, y in coords:
        axis.annotate(
            f"$({x}, {y:.8f})$",
            xy=(x, y),
            xytext=None,
            color="white",
            fontsize=12,
            arrowprops=dict(facecolor=Color.Dark.ORANGE, shrink=0.1),
        )

    axis.grid(color=Color.Dark.GRAY, linestyle="--", linewidth=0.8)

    title = "LR Scheduler"
    if lr_scheduler_type:
        title += f": {lr_scheduler_type}"

    plt.title(title)
    plt.tight_layout()
    plt.show()
