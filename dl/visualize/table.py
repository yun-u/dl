import functools
from dataclasses import asdict
from typing import List, Literal, Optional

import numpy as np

from .style import Style, colored_text


class Cell:
    def __init__(
        self,
        row: int,
        col: int,
        text: str,
        min_width: int = 3,
        vertial_align: Literal["<", "^", ">"] = "^",
        style: Optional[Style] = None,
    ) -> None:
        self.row = row
        self.col = col
        self.text = text
        self.min_width = min_width
        self.width = len(text)
        self.vertical_align = vertial_align
        self.style = style

    def render(self, width: int) -> str:
        text = f"{self.text:{self.vertical_align}{max(width, self.min_width)}}"
        style = {} if self.style is None else asdict(self.style)
        return colored_text(text, **style)

    def __repr__(self) -> str:
        return f"Cell(row={self.row}, col={self.col}, text={self.text})"


class Table:
    def __init__(self, title: str = "") -> None:
        self.title = title

        self.cells: List[Cell] = []
        self.data: Optional[np.ndarray] = None

    def add_cell(self, cell: Cell):
        self.cells.append(cell)

    def construct(self):
        row = max([c.row for c in self.cells]) + 1
        col = max([c.col for c in self.cells]) + 1

        self.data = np.empty((row, col), dtype=object)

        for c in self.cells:
            self.data[c.row, c.col] = c

        ys, xs = np.nonzero(self.data == None)  # noqa: E711
        for y, x in zip(ys, xs):
            self.data[y, x] = Cell(row=y, col=x, text="")

        assert (self.data != None).all()  # noqa: E711

    @functools.cached_property
    def text_widths(self) -> np.ndarray:
        assert self.data is not None
        return np.vectorize(lambda x: len(x.text))(self.data)

    def render(self) -> str:
        if self.data is None:
            self.construct()

        texts = np.vectorize(lambda cell, width: cell.render(width))(
            self.data, self.text_widths.max(axis=0)
        )

        renderd_texts = []

        if self.title:
            renderd_texts.append(colored_text(self.title, italic=True) + "\n")

        renderd_texts.extend(["".join(row) for row in texts])
        return "\n".join(renderd_texts)
