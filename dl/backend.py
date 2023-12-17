from functools import singledispatch
from typing import Optional, Tuple, Union, overload

import numpy as np
import torch

# https://github.com/microsoft/pyright/issues/988#issuecomment-867148941


@singledispatch
def _dim(x):
    raise NotImplementedError


@_dim.register
def _(x: np.ndarray) -> int:
    return x.ndim


@_dim.register
def _(x: torch.Tensor) -> int:
    return x.dim()


@overload
def dim(x: torch.Tensor) -> int:
    ...


@overload
def dim(x: np.ndarray) -> int:
    ...


def dim(*args, **kwargs):
    return _dim(*args, **kwargs)


@singledispatch
def _unsqueeze(x):
    raise NotImplementedError


@_unsqueeze.register
def _(x: np.ndarray, dim: int) -> np.ndarray:
    return np.expand_dims(x, axis=dim)


@_unsqueeze.register
def _(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x.unsqueeze(dim)


@overload
def unsqueeze(x: torch.Tensor) -> int:
    ...


@overload
def unsqueeze(x: np.ndarray) -> int:
    ...


def unsqueeze(*args, **kwargs):
    return _unsqueeze(*args, **kwargs)


@singledispatch
def _size(x):
    raise NotImplementedError


@_size.register
def _(x: np.ndarray, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
    if dim:
        return x.shape[dim]
    return x.shape


@_size.register
def _(x: torch.Tensor, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
    if dim:
        return x.size(dim)
    return tuple(x.size())


@overload
def size(x: torch.Tensor, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
    ...


@overload
def size(x: np.ndarray, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
    ...


def size(*args, **kwargs):
    return _size(*args, **kwargs)
