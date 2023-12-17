import json
import math
import os
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import mlflow
import numpy as np
import torch
from safetensors.torch import save_model


def is_scalar(value: Any) -> bool:
    """Check if a value is a scalar (int or float).

    Args:
        value (`Any`): The value to check.

    Returns:
        `bool`: True if the value is a scalar, False otherwise.
    """
    return isinstance(value, (int, float))


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""

    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(self, obj)


class Callback:
    """Base class for callbacks"""

    def on_batch_end(self, batch: int, log: Dict[str, Any]) -> None:
        """Callback function called at the end of each batch.

        Args:
            batch (`int`): The current iteration number.
            log (`Dict[str, Any]`): A dictionary containing logged information.
        """
        pass

    def on_epoch_end(self, epoch: int, log: Dict[str, Any]) -> None:
        """Callback function called at the end of each epoch.

        Args:
            epoch (`int`): The current epoch number.
            log (`Dict[str, Any]`): A dictionary containing logged information.
        """
        pass


class History(Callback):
    """Callback for logging training history to a file

    Args:
        path (`Union[str, PathLike]`): The path to save the history file.
    """

    def __init__(self, path: Union[str, PathLike]) -> None:
        self.path = path

    def append_log(self, log: Dict[str, Any]) -> None:
        """Append logged information to the specified log file.

        Args:
            log (`Dict[str, Any]`): A dictionary containing logged information.
        """
        with open(self.path, "a") as f:
            f.write(json.dumps(log, cls=NumpyEncoder) + "\n")

    def on_batch_end(self, batch: int, log: Dict[str, Any]) -> None:
        self.append_log(log)

    def on_epoch_end(self, epoch: int, log: Dict[str, Any]) -> None:
        self.append_log(log)


class BestMetric(Callback):
    """Callback for saving the best model based on a specified metric

    Args:
        root (`Union[str, PathLike]`): Root directory to save the model.
        model (`torch.nn.Module`): The PyTorch model to save.
        metric_name (`Union[str, Sequence[str]]`): The name of the metric to monitor.
        greater_is_better: (`bool`): The decision based on monitored quantity.
        filename (`Union[str, PathLike]`): The filename to save the model.
        save_method (`Callable[[torch.nn.Module, str], None]`): The method for saving models.
    """

    def __init__(
        self,
        root: Union[str, PathLike],
        model: torch.nn.Module,
        metric_name: Union[str, Sequence[str]],
        greater_is_better: bool,
        filename: Union[
            str, PathLike
        ] = "epoch:{epoch:03d}-{metric}:{value}.safetensors",
        save_method: Callable[[torch.nn.Module, str], Any] = save_model,
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.metric_name = (metric_name,) if isinstance(metric_name, str) else tuple(metric_name)  # fmt: skip
        self.greater_is_better = greater_is_better
        self.filename = filename
        self.save_method = save_method

        self.op = max if greater_is_better else min
        self.best_value = [-math.inf if greater_is_better else +math.inf] * len(self.metric_name)  # fmt: skip
        self.best_epoch = -1
        self.best_model_path: Optional[str] = None

    def save_best_model(self) -> None:
        """Save the best model based on the monitored metric."""

        # remove previous best model if it exists
        if self.best_model_path and Path(self.best_model_path).exists():
            os.remove(self.best_model_path)

        self.best_model_path = str(
            self.root
            / self.filename.format(
                epoch=self.best_epoch,
                metric=str([n.replace("/", "@") for n in self.metric_name]),
                value=str([f"{v:g}" for v in self.best_value]),
            )
        )

        self.save_method(self.model, self.best_model_path)

    def on_epoch_end(self, epoch: int, log: Dict[str, Any]) -> None:
        metric_values = []

        for name in self.metric_name:
            if (metric_value := log.get(name)) is None:
                continue

            if not is_scalar(metric_value):
                continue

            metric_values.append(metric_value)

        if len(metric_values) == 0:
            return

        mean_value = np.mean(metric_values)

        if self.op(np.mean(self.best_value), mean_value) == mean_value:
            self.best_value = metric_values
            self.best_epoch = epoch
            self.save_best_model()


class Mlflow(Callback):
    """Callback for logging to Mlflow"""

    def on_batch_end(self, batch: int, log: Dict[str, Any]) -> None:
        mlflow.log_metrics({k: v for k, v in log.items() if is_scalar(v)}, step=batch)

    def on_epoch_end(self, epoch: int, log: Dict[str, Any]) -> None:
        mlflow.log_metrics(
            {f"epoch/{k}": v for k, v in log.items() if is_scalar(v)}, step=epoch
        )
