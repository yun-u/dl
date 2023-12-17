from os import PathLike
from typing import (
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Integer, UInt8
from numpy.typing import ArrayLike
from skimage import measure
from torchvision.utils import make_grid

from dl.utils.backend import BaseTensor, detect_tensor_format, dim, size, unsqueeze

from .style import Color, Style
from .table import Cell, Table

################################################################################
# Visualize Attention
################################################################################


def divide_to_patch(image: BaseTensor, patch_size: int) -> BaseTensor:
    if dim(image) != 3:
        raise RuntimeError(f"Unexpected dimension size {dim(image)}")

    if (tensor_format := detect_tensor_format(unsqueeze(image, 0))) == "NHWC":
        image = rearrange(image, "h w c -> c h w")

    h, w = size(image, 1), size(image, 2)
    num_h_patches, num_w_patches = h // patch_size, w // patch_size

    image = image[:, : num_h_patches * patch_size, : num_w_patches * patch_size]

    patches = rearrange(
        image,
        "c (nhp ph) (nwp pw) -> nhp nwp c ph pw",
        nhp=num_h_patches,
        nwp=num_w_patches,
        ph=patch_size,
        pw=patch_size,
    )

    if tensor_format == "NHWC":
        patches = rearrange(patches, "nhp nwp c ph pw -> nhp nwp ph pw c")

    return patches


def visualize_patch(
    image: BaseTensor, patch_size: int, pad_value: int = 0
) -> torch.Tensor:
    if detect_tensor_format(unsqueeze(image, 0)) == "NHWC":
        image = rearrange(image, "h w c -> c h w")

    patches = divide_to_patch(image, patch_size)
    num_w_patches = size(patches, 1)

    if isinstance(patches[0], np.ndarray):
        patches = torch.from_numpy(patches.copy())

    return make_grid(
        rearrange(patches, "nhp nwp c ph pw -> (nhp nwp) c ph pw"),
        nrow=num_w_patches,
        pad_value=pad_value,
    )


def visualize_attention(
    attentions: Float[torch.Tensor, "HEAD HW"], patch_size: int, image: torch.Tensor
) -> torch.Tensor:
    if attentions.dim() != 2:
        raise RuntimeError(f"Unexpected dimension size {attentions.dim()}")

    h, w = image.size(1), image.size(2)
    num_h_patches, num_w_patches = h // patch_size, w // patch_size

    image = image[:, : num_h_patches * patch_size, : num_w_patches * patch_size]

    attentions = rearrange(
        attentions, "n_heads (h w) -> n_heads h w", h=num_h_patches, w=num_w_patches
    )
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )
    return make_grid(attentions, normalize=True, scale_each=True)


################################################################################
# Visualize Confusion Matrix
################################################################################


def display_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    display_labels: Optional[ArrayLike] = None,
    normalize: Optional[Literal["true", "pred", "all"]] = None,
    cmap: str = "Blues",
    save_path: Union[str, PathLike, None] = None,
):
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=display_labels,
        cmap=getattr(plt.cm, cmap),
        normalize=normalize,
    )
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def display_text_confusion_matrix(
    y_true: Optional[ArrayLike],
    y_pred: Optional[ArrayLike],
    confusion_matrix: Union[np.ndarray, None] = None,
    title: str = "",
    cmap: str = "Blues",
):
    """
    Plots a confusion matrix given the true labels and predicted labels.

    Args:
        y_true (`List`): List of true labels.
        y_pred (`List`): List of predicted labels.
        cmap (`str`, optional): Colormap to use for the matrix. Defaults to "Blues".
    """
    if not (
        (y_true is not None and y_pred is not None) ^ (confusion_matrix is not None)
    ):
        raise ValueError

    if y_true is not None and y_pred is not None:
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    elif confusion_matrix is not None:
        cm = confusion_matrix

    norm = mpl.colors.Normalize(vmin=cm.min(), vmax=cm.max())

    n_classes = cm.shape[0]
    n_trues, n_preds = cm.sum(axis=1), cm.sum(axis=0)

    table = Table(title=title)

    # add the top-left cell as "T\P"
    table.add_cell(Cell(row=0, col=0, text="T\P", style=Style(bold=True)))

    # add class labels as column headers; y_pred
    for i in range(0, n_classes):
        reverse = False if i % 2 == 0 else True
        table.add_cell(
            Cell(
                row=0,
                col=i + 1,
                text=str(i),
                style=Style(color=(152, 152, 157), bold=True, reverse=reverse),
            )
        )

    # iterate over the confusion matrix and add cells
    for i in range(n_classes):
        # add class labels as first column; y_true
        reverse = False if i % 2 == 0 else True
        table.add_cell(
            Cell(
                row=i + 1,
                col=0,
                text=str(i),
                style=Style(color=(152, 152, 157), bold=True, reverse=reverse),
            )
        )

        for j in range(n_classes):
            value = cm[i, j]
            background_color = getattr(plt.cm, cmap)(norm(value), bytes=True)[:3]
            table.add_cell(
                Cell(
                    row=i + 1,
                    col=j + 1,
                    text=str(value),
                    style=Style(
                        color=(142, 142, 147), background_color=background_color
                    ),
                )
            )

        # add the sum of true labels for the class
        table.add_cell(
            Cell(
                row=i + 1,
                col=j + 2,
                text=f"{n_trues[i]}",
                min_width=2,
                vertial_align=">",
                style=Style(color=(48, 209, 88), bold=True, reverse=reverse),
            )
        )

    # add the sum of predicted labels for each class as a row at the bottom
    for j in range(n_classes):
        reverse = False if j % 2 == 0 else True
        table.add_cell(
            Cell(
                row=n_classes + 1,
                col=j + 1,
                text=f"{n_preds[j]}",
                vertial_align="^",
                style=Style(color=(48, 209, 88), bold=True, reverse=reverse),
            )
        )

    # render and print the table
    print(table.render())


################################################################################
# Visualize Bounding Box & Segmentation
################################################################################


def random_color_iterator(n: int) -> Iterator[str]:
    from sklearn.cluster import KMeans

    xs = np.linspace(0, 255, 8)
    xv, yv, zv = np.meshgrid(xs, xs, xs)
    X = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto", max_iter=100).fit(X)

    for rgb in sorted(np.uint8(kmeans.cluster_centers_), key=lambda x: x.sum()):
        yield "#" + "".join([f"{i:02X}" for i in rgb])


def draw_segmentation_masks(
    image: UInt8[np.ndarray, "H W"],
    masks: Union[List[Integer[np.ndarray, "H W"]], Integer[np.ndarray, "N H W"]],
):
    import plotly.express as px

    contours = []
    for mask in masks:
        assert mask.ndim == 2, mask.shape
        contour = measure.find_contours(mask, 0.5)[0]
        contours.append(contour)

    random_color = random_color_iterator(len(masks))

    fig = px.imshow(image)
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    for i, contour in enumerate(contours):
        y, x = contour.T

        fig.add_scatter(
            x=x,
            y=y,
            name=str(i),
            mode="lines",
            opacity=0.6,
            fill="toself",
            fillcolor=next(random_color),
            line={"color": "white"},
            showlegend=False,
            hoveron="fills",
        )

    # specifying themes: https://plotly.com/python/templates/ https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-label-properties/app.py
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), template="plotly_dark")
    fig.update_yaxes(visible=False, range=[image.shape[0], 0])
    fig.update_xaxes(visible=False, range=[0, image.shape[1]])
    fig.show()


def draw_bounding_boxes(
    image: UInt8[np.ndarray, "H W"],
    boxes: Integer[np.ndarray, "N 4"],
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
):
    import plotly.express as px

    random_color = random_color_iterator(len(boxes))

    fig = px.imshow(image)
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    colors = []
    if labels:
        assert len(boxes) == len(labels)
        label_dict = {}
        for label in set(labels):
            label_dict[label] = next(random_color)

        for label in labels:
            colors.append(label_dict[label])
    else:
        for _ in range(len(boxes)):
            colors.append(next(random_color))
        labels = [str(i) for i in range(len(boxes))]

    if scores is None:
        str_scores = [""] * len(boxes)
    else:
        str_scores = [f"{score * 100:.2f}%" for score in scores]

    for box, label, color, score in zip(boxes, labels, colors, str_scores):
        xmin, ymin, xmax, ymax = box

        fig.add_scatter(
            x=[xmin, xmin, xmax, xmax, xmin],
            y=[ymin, ymax, ymax, ymin, ymin],
            name=label,
            mode="lines+text",
            text=[score],
            line={"color": color, "width": 2},
            textposition="bottom right",
            textfont={"color": "white", "family": "Courier New", "size": 18},
            showlegend=True,
        )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), template="plotly_dark")
    fig.update_yaxes(visible=False, range=[image.shape[0], 0])
    fig.update_xaxes(visible=False, range=[0, image.shape[1]])
    fig.show()


################################################################################
# EDA
################################################################################


def visualize_aspect_ratio(shapes: List[Tuple[int, int]]):
    """
    Visualize aspect ratios of shapes.

    Args:
        shapes: A list of tuples representing the height and width of shapes.
    """
    heights, widths = mit.transpose(shapes)

    aspect_ratios = [w / h for (h, w) in shapes]

    argminr, argmaxr = np.argmin(aspect_ratios), np.argmax(aspect_ratios)
    minr, maxr = aspect_ratios[argminr], aspect_ratios[argmaxr]

    hmin, hmax = mit.minmax(heights)
    wmin, wmax = mit.minmax(widths)

    fig, axis = plt.subplots(nrows=1, ncols=1)

    axis.hexbin(heights, widths, cmap="coolwarm", gridsize=50, linewidths=(0,))
    axis.axline(
        xy1=(hmin, wmin),
        xy2=(hmax, wmax),
        color=Color.GRAY12,
        linestyle="dotted",
        alpha=0.8,
    )
    axis.axline(
        xy1=(hmin, hmin * minr),
        xy2=(hmax, hmax * minr),
        color=Color.GRAY12,
        linestyle="dotted",
        alpha=0.8,
    )
    axis.axline(
        xy1=(hmin, hmin * maxr),
        xy2=(hmax, hmax * maxr),
        color=Color.GRAY12,
        linestyle="dotted",
        alpha=0.8,
    )

    for arg in [argminr, argmaxr]:
        h, w = heights[arg], widths[arg]
        axis.scatter(h, w, s=30, c=Color.GRAY12, marker="*")
        axis.text(
            h,
            w,
            f" ({h:.0f}, {w:.0f}) [{w/h:.2f}]",
            color=Color.Light.ORANGE,
            fontfamily="monospace",
            fontweight="bold",
            ha="left",
            va="center",
        )

    axis.set_xlim(hmin, hmax)
    axis.set_ylim(wmin, wmax)

    axis.set_aspect("equal", "box")
    axis.set_xlabel("height")
    axis.set_ylabel("width")

    plt.title("Aspect Ratio")
    plt.tight_layout()
    plt.show()
