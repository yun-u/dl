from __future__ import annotations

import pprint
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import more_itertools as mit
import numpy as np
import torch
from loguru._colorizer import Colorizer
from tqdm import tqdm

from dl.logging_setup import get_logger
from dl.visualize.utils import display_text_confusion_matrix

from .callback import Callback
from .metric import Metric, is_metric_fn
from .utils import transpose

__all__ = [
    "Log",
    "Logger",
    "LogValue",
]

logger = get_logger()


@dataclass
class LogValue:
    """Represents a logged value with its associated weight.

    Args:
        value (`Any`): The logged value. In most cases, the value is corresponding to one batch.
        weight (`Union[int, float]`): The weight assigned to the value (default: 1).
    """

    value: Any
    weight: Union[int, float] = 1


def add_prefix(prefix: str, log: Dict[str, Any]) -> Dict[str, Any]:
    """Add a prefix to the keys in a dictionary.

    Args:
        prefix (`str`): The prefix to add.
        log (`Dict[str, Any]`): The input dictionary.

    Returns:
        `Dict[str, Any]`: A new dictionary with keys containing the prefix.
    """
    if prefix == "":
        return log

    return {f"{prefix}/{key}": value for key, value in log.items()}


class Log:
    """A log object with an optional prefix.

    Attributes:
        records (`Dict[int, Dict[str, LogValue]]`): A dictionary to store log
            with iteration index representing indices and values as dictionaries
            containing log entries with names as keys.
    """

    def __init__(self) -> None:
        self.records: Dict[int, Dict[str, LogValue]] = {}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get log record for a specific index.

        Args:
            index (`int`): The iteration index of the log record.

        Returns:
           `Dict[str, Any]`: Log record for the specified index.
        """
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        """Return a string representation of the log object."""
        return pprint.pformat({"records": self.records}, indent=2)

    @staticmethod
    def convert_type(item: Tuple[str, Any]) -> Tuple[str, Any]:
        """Convert certain types in log items for consistent formatting.

        | type         | converted type |
        |:-------------|:---------------|
        | torch.Tensor | np.numpy       |
        | Sequence     | np.numpy       |
        | int, float   | int, float     |
        | Any          | Any            |

        Args:
            item (`Tuple[str, Any]`): A key-value pair.

        Returns:
            `Tuple[str, Any]`: The converted key-value pair.
        """
        k, v = item

        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().clone().numpy()
            return Log.convert_type((k, v))

        if isinstance(v, np.ndarray) and len(v.shape) == 0:
            return k, v.item()

        if isinstance(v, Sequence):
            return k, np.asarray(v)

        return k, v

    def collect(self, index: int, weight: int = 1, **kwargs):
        """Collect log record for a specific index and prefix.

        Args:
            index (`int`): The index to collect record for.
            weight (`int`): The weight associated with the log record.
            **kwargs: Key-value pairs to collect as log record.
        """
        assert index not in self.records

        log = [self.convert_type((k, v)) for k, v in kwargs.items() if v is not None]

        self.records[index] = {k: LogValue(v, weight) for k, v in log}

    def last_index(self) -> int:
        """Get the index of the last collected log record.

        Returns:
            `int`: The index of the last log record, or -1 if no record is available.
        """
        if len(self.records) == 0:
            return -1
        return max(self.records.keys())

    def last(self, exclude: Union[set[str], None] = None) -> Dict[str, Any]:
        """Get the most latest log record, excluding specified keys.

        Args:
            exclude (`Union[set[str], None]`): A set of keys to exclude from the log record.
                If None, no keys are excluded.

        Returns:
            `Dict[str, Any]`: The most recent log record with excluded keys removed.
        """
        assert len(self.records) > 0

        exclude = exclude or set()

        index = self.last_index()

        record = self.records[index]

        # exclude keys from the log record
        return {k: v.value for k, v in record.items() if k not in exclude}

    def merge(
        self, metrics: Sequence[Metric] = (), exclude: Union[set[str], None] = None
    ) -> Dict[str, Any]:
        """Merge collected log record and compute metrics.

        Args:
            metrics (`Sequence[Metric]`): Sequence of metric functions to compute.
            exclude (`Union[set[str], None]`): A set of keys to exclude from the log record.
                If None, no keys are excluded.

        Returns:
            `Dict[str, Any]`: Merged log record with metrics.
        """
        exclude = exclude or set()

        # sort by iteration index order
        records = [self.records[i] for i in sorted(self.records.keys())]

        # change the key from iteration index to log name
        records: Dict[str, List[LogValue]] = transpose(records)

        merged: Dict[str, Any] = {}
        for key, log_values in records.items():
            values, weights = mit.transpose(
                [(log_value.value, log_value.weight) for log_value in log_values]
            )

            if isinstance(values[0], np.ndarray):
                merged[key] = np.concatenate(values, axis=0)
            elif isinstance(values[0], (int, float)):
                merged[key] = np.average(values, weights=weights)
            else:
                merged[key] = values

        # as in predicting a testset, if label information is not given, the metric is not evaluated.
        if {"labels", "logits"} <= merged.keys():
            merged.update(
                self.compute_metrics(
                    metrics, **merged
                )  # merged["labels"], merged["logits"])
            )

        # exclude keys from the log record
        return {k: v for k, v in merged.items() if k not in exclude}

    def compute_metrics(
        self,
        metrics: Sequence[Metric],
        labels: np.ndarray,
        logits: np.ndarray,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Compute metrics and add them to the log record.

        Args:
            metrics (`Sequence[Metric]`): Sequence of metric functions to compute.
            labels (`np.ndarray`): The ground truth labels.
            logits (`np.ndarray`): The predicted logits.

        Returns:
            `Dict[str, np.ndarray]`: A dictionary of computed metrics.
        """
        if len(metrics) == 0:
            return {}

        return {metric.__name__: metric(labels, logits, **kwargs) for metric in metrics}


DataSplit = Literal["train", "val", "test"]
Prefix = Union[str, DataSplit]

LOG_TEXT_FORMAT = {
    "bold": "<white><bold>{}</></>",
    "iter": "<magenta><bold>{}</></>",
    "epoch": "<red><bold>{}</></>",
    "number": "<cyan><bold>{}</></>",
}


class Logger:
    """A logger for tracking training progress.

    Args:
        max_iters (`int`): Maximum number of iterations.
        metrics (`List[Metric]`): List of metric functions to track.
        callbacks (`List[Callback]`): List of callback.
    """

    def __init__(
        self, max_iters: int, metrics: List[Metric], callbacks: List[Callback]
    ) -> None:
        self.max_iters = max_iters
        self.metrics: List[Metric] = []
        self.callbacks: List[Callback] = []

        self.logs: Dict[Prefix, Log] = {}

        self.batch_progress_bar = tqdm(
            total=None if max_iters < 0 else max_iters, leave=False
        )

        self.add_metrics(*metrics)
        self.add_callbacks(*callbacks)

    def add_metric(self, metric: Metric) -> None:
        """Add a metric function to track during training.

        Args:
            metric (`Metric`): The metric function to add.
        """
        if not is_metric_fn(metric):
            raise ValueError("The provided metric is not a valid metric function.")

        self.metrics.append(metric)

    def add_metrics(self, *metrics: Metric) -> None:
        """Add a metric function to track during training.

        Args:
            *metrics (`Metric`): The metric function to add.
        """
        for metric in metrics:
            self.add_metric(metric)

    def add_callback(self, callback: Callback) -> None:
        """Add a callback.

        Args:
            callback (`Callback`): The callback to add.
        """
        if not isinstance(callback, Callback):
            raise ValueError("The provided callback is not a valid callback.")

        self.callbacks.append(callback)

    def add_callbacks(self, *callbacks: Callback) -> None:
        """Add a callback.

        Args:
            *callbacks (`Callback`): The callback to add.
        """
        for callback in callbacks:
            self.add_callback(callback)

    def prepare_log(
        self,
        prefix: Union[Prefix, Tuple[Prefix, ...]],
        log_type: Literal["batch", "epoch"],
        exclude: Union[set[str], None] = {"labels", "logits"},
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare log record for printing and logging.

        Args:
            prefix (`Union[Prefix, Tuple[Prefix, ...]]`): The log prefix or prefixes.
            log_type (`Literal["batch", "epoch"]`): The phase of log record (batch or epoch).
                "batch" if evaluated before iterating the entire train dataset. "epoch" if evaluated after iteration.
            **kwargs: Additional log record to include.

        Returns:
            `Dict[str, Any]`: Prepared log record.

        | log_type | prefix         | Action                                                    |
        |:---------|:---------------|:----------------------------------------------------------|
        | batch    | train          | Return the most recent iteration.                         |
        | batch    | val, test, ... | Merge (pop and merge) log record for val or test prefix. |
        | epoch    | train          | Merge (pop and merge) train log record.                   |
        | epoch    | val, test, ... | Merge (pop and merge) log record for val or test prefix. |
        """
        if log_type == "epoch" and "epoch" not in kwargs:
            raise ValueError("The 'epoch' key is missing from the provided log record.")

        exclude = exclude or set()

        if isinstance(prefix, tuple):
            cond = zip(prefix, (log_type,) * len(prefix))
        else:
            cond = [(prefix, log_type)]

        prepared_log = {}
        for prefix, log_type in cond:
            if prefix not in self.logs:
                continue

            if log_type == "batch":
                log = self.logs[prefix]
                prepared_log.update({"iter": log.last_index()})

            if (prefix, log_type) == ("train", "batch"):
                log = self.logs[prefix]
                prepared_log.update(add_prefix(prefix, log.last(exclude)))
            else:
                log = self.logs.pop(prefix)
                prepared_log.update(
                    add_prefix(prefix, log.merge(self.metrics, exclude))
                )

        return {**prepared_log, **kwargs}

    def handle_callback(
        self, log_type: Literal["batch", "epoch"], log: Dict[str, Any]
    ) -> None:
        for callback in self.callbacks:
            if log_type == "batch":
                callback.on_batch_end(log["iter"], log)
            elif log_type == "epoch":
                callback.on_epoch_end(log["epoch"], log)

    def log(
        self,
        prefix: Union[Prefix, Tuple[Prefix, ...]],
        log_type: Literal["batch", "epoch"],
        **kwargs,
    ) -> Dict[str, Any]:
        """Log and print log record.

        Args:
            prefix (`Union[Prefix, Tuple[Prefix, ...]]`): The log prefix or prefixes.
            log_type (`Literal["batch", "epoch"]`): The type of log record (batch or epoch).
            **kwargs: Additional log record to include.

        Returns:
            `Dict[str, Any]`: The logged and prepared log record.
        """
        log = self.prepare_log(prefix, log_type, **kwargs)
        log = {k: log[k] for k in sorted(log.keys() - {"labels", "logits"})}

        self.print(prefix, log_type, log)
        self.handle_callback(log_type, log)

        return log

    def collect(
        self,
        index: int,
        prefix: Prefix,
        weight: int = 1,
        **kwargs,
    ):
        """Collect log record for a specific index and prefix.

        Args:
            index (`int`): The index to collect record for.
            prefix (`Prefix`): The log prefix.
            weight (`int`): The weight associated with the log record.
            **kwargs: Key-value pairs to collect as log record.
        """
        log = self.logs.setdefault(prefix, Log())
        log.collect(index, weight, **kwargs)

    @staticmethod
    def has_prefix(
        prefix: Union[Prefix, Tuple[Prefix, ...]], prefix_type: Prefix
    ) -> bool:
        """Check if a given prefix or prefixes match a specific type.

        Args:
            prefix (`Union[Prefix, Tuple[Prefix, ...]]`): The log prefix or prefixes.
            prefix_type (`Prefix`): The type to check against.

        Returns:
            `bool`: True if there is a match, False otherwise.
        """
        if isinstance(prefix, str) and prefix == prefix_type:
            return True

        if isinstance(prefix, tuple) and prefix_type in prefix:
            return True

        return False

    def print(
        self,
        prefix: Union[Prefix, Tuple[Prefix, ...]],
        log_type: Literal["batch", "epoch"],
        log: Dict[str, Any],
    ):
        """Print log record.

        Args:
            prefix (`Union[Prefix, Tuple[Prefix, ...]]`): The log prefix or prefixes.
            log_type (`Literal["batch", "epoch"]`): The type of log record (batch or epoch).
            log (`Dict[str, Any]`): The log record to print.
        """
        message = self.format(log)

        if self.has_prefix(prefix, "train") and log_type == "batch":
            self.batch_progress_bar.update(1)
            self.batch_progress_bar.set_description(
                Colorizer.prepare_format(message).colorize(0)
            )

        logger.opt(colors=True).info(message)

        for prefix in ["train", "val", "test"]:
            if (cm := log.get(f"{prefix}/confusion_matrix")) is not None:
                display_text_confusion_matrix(
                    y_true=None,
                    y_pred=None,
                    confusion_matrix=cm,
                    title=f"Confusion Matrix: {prefix}, {log_type}",
                )

    def get_iter_and_epoch_message(
        self, iter_: Optional[int], epoch: Optional[int]
    ) -> str:
        """Generate a message for the current iteration and epoch.

        Args:
            iter_ (`int`, optional): The current iteration number.
            epoch (`int`, optional): The current epoch number.

        Returns:
            `str`: A formatted message containing the iteration and/or epoch information.
        """
        message = ""

        if iter_ is not None:
            message += LOG_TEXT_FORMAT["iter"].format(
                f"Step [{iter_}/{self.max_iters}] "
            )
        if epoch is not None:
            message += LOG_TEXT_FORMAT["epoch"].format(f"Epoch [{epoch}] ")

        return message

    def format(self, log: Dict[str, Any]) -> str:
        """Format log record as a string for printing.

        Args:
            log (`Dict[str, Any]`): The log record to format.

        Returns:
            `str`: A formatted log string.
        """
        iter_and_epoch = self.get_iter_and_epoch_message(
            log.get("iter"), log.get("epoch")
        )

        messages = []
        for k in sorted(log.keys() - {"iter", "epoch"}):
            v = log[k]

            if isinstance(v, np.ndarray) and v.size > 1:
                continue

            if isinstance(v, np.ndarray) and v.size == 1:
                v = f"{v.item():g}"
            elif isinstance(v, float):
                v = f"{v:g}"
            else:
                v = str(v)

            k = LOG_TEXT_FORMAT["bold"].format(k)
            v = LOG_TEXT_FORMAT["number"].format(v)

            messages.append(f"{k}: {v}")

        return iter_and_epoch + ", ".join(messages)
