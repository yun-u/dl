import warnings
from os import PathLike
from pathlib import Path
from typing import Literal, Tuple, Union

import cv2
import numpy as np
import torch
from einops import rearrange
from jaxtyping import AbstractDtype, UInt8
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .backend import detect_tensor_format


class ImageDtype(AbstractDtype):
    dtypes = ["uint8", "float32", "float64"]


def read_image(
    path: Union[str, PathLike], format: Literal["RGB", "BGR"] = "RGB"
) -> UInt8[np.ndarray, "H W"]:
    if isinstance(path, Path):
        path = str(path)

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if format == "BGR":
        return image
    elif format == "RGB":
        # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image[..., ::-1]
    else:
        RuntimeError(
            "Invalid image format specified. Supported formats are 'RGB' and 'BGR'."
        )


def normalize(
    x: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    return standardize(x, mean, std, normalize=True)


def unnormalize(
    x: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    return standardize(x, mean, std, normalize=False)


def standardize(
    x: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    normalize: bool = True,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Unexpected dimension size {x.dim()}")

    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)

    if detect_tensor_format(x) == "NCHW":
        mean, std = rearrange(mean, "c -> 1 c 1 1"), rearrange(std, "c -> 1 c 1 1")

    if normalize:
        x = (x - mean) / std
    else:
        x = x * std + mean

    if x.max() > 1.0 or x.min() < 0.0:
        warnings.warn(
            " The expected range of values for the input is from 0.0 to 1.0 Please ensure that your input values are within this range to ensure correct processing."
        )

    return x


def to_pil_image(x: torch.Tensor) -> Image.Image:
    if x.dim() != 3:
        raise ValueError(f"Unexpected dimension size {x.dim()}")

    if detect_tensor_format(x.unsqueeze(0)) == "NCHW":
        x = rearrange(x, "c h w -> h w c")

    if x.min() < 0 or x.max() > 255:
        warnings.warn(
            "The input image contains pixel values outside the valid range of 0-255. These values will be clipped to the valid range during processing."
        )

    x = torch.clip(x, min=0, max=255)
    return Image.fromarray(x.type(torch.uint8).numpy())
