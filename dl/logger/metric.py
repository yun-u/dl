from __future__ import annotations

import inspect
from inspect import Parameter
from typing import (
    Any,
    Callable,
)

import numpy as np
from typing_extensions import Annotated

Metric = Callable[
    [Annotated[np.ndarray, "labels"], Annotated[np.ndarray, "logits"]], Any
]


def is_metric_fn(fn: Metric) -> bool:
    """Checks if a function meets the criteria of a metric function.

    Args:
        fn (`Metric`): The function to be checked.

    Returns:
        `bool`: True if the function is a metric function, False otherwise.
    """
    sig = inspect.signature(fn)
    params = [
        {"name": name, "param": param} for (name, param) in sig.parameters.items()
    ]

    if not (
        len(params) >= 2
        and params[0]["name"] == "labels"
        and params[1]["name"] == "logits"
    ):
        return False

    for param in params[2:]:
        p: inspect.Parameter = param["param"]
        if p.default == inspect._empty and p.kind not in (
            Parameter.VAR_KEYWORD,
            Parameter.VAR_POSITIONAL,
        ):
            return False
    return True
