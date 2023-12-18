from PIL import Image  # type: ignore
from abc import abstractmethod
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union
from typing_extensions import Self

import torch
import numpy as np

Array = Union[np.ndarray, torch.Tensor]
Number = Union[int, float]
Vertex = Tuple[Number, Number]  # (x, y)
Box = Tuple[Vertex, Vertex, Vertex, Vertex]  # ((x0, y0), (x1, y1), (x2, y2), (x3, y3))
UiInfo = Dict[str, Any]


class BBox(NamedTuple):
    v0: Vertex
    v1: Vertex

    @classmethod
    def new(cls, x0: Number, y0: Number, x1: Number, y1: Number) -> Self:
        return cls((x0, y0), (x1, y1))

    def flatten(self) -> Tuple[Number, Number, Number, Number]:
        return *self.v0, *self.v1

    def scale(self, scale_x: Number, scale_y: Number) -> Self:
        (x0, y0), (x1, y1) = self
        return self.__class__(
            (x0 * scale_x, y0 * scale_y), (x1 * scale_x, y1 * scale_y)
        )

    def map(self, func: Callable[[Number], Number]) -> Self:
        (x0, y0), (x1, y1) = self
        return self.__class__((func(x0), func(y0)), (func(x1), func(y1)))

    def to_int_flattened(self) -> Tuple[int, int, int, int]:
        (x0, y0), (x1, y1) = self
        return int(x0), int(y0), int(x1), int(y1)


class UiElement:
    name: str
    bbox: BBox
    screenshot: Union[str, Array, None]
    info: UiInfo

    def __init__(
        self,
        name: str,
        bbox: BBox,
        screenshot: Optional[Union[str, Array]] = None,
        info: Optional[UiInfo] = None,
    ):
        """
        Parameters:
        ----------
        name : str
            name of the UI class
        bbox : BBox
            bounding box of the UI element
        screenshot : np.ndarray
            the original screenshot of shape (h, w, c)
        """
        self.name = name
        self.bbox = bbox
        self.screenshot = screenshot
        self.info = {} if info is None else info

    @classmethod
    def from_xywh(
        cls,
        name: str,
        xywh: Tuple[Number, Number, Number, Number],
        screenshot: Array,
        is_center: bool = False,
    ) -> Self:
        x, y, w, h = xywh
        if is_center:
            x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        else:
            (
                x0,
                y0,
                x1,
                y1,
            ) = (
                x,
                y,
                x + w,
                y + h,
            )
        return cls(name, BBox((x0, y0), (x1, y1)), screenshot)

    def size(self) -> Tuple[Number, Number]:
        (x0, y0), (x1, y1) = self.bbox
        return abs(x0 - x1), abs(y0 - y1)

    def get_cropped_image(self, loader: Optional[Callable[..., Array]] = None) -> Array:
        x0, y0, x1, y1 = self.bbox.to_int_flattened()
        return self.get_screenshot(loader)[y0:y1, x0:x1]

    def get_screenshot(self, loader: Optional[Callable[..., Array]] = None) -> Array:
        if self.screenshot is None:
            if loader is None:
                raise ValueError("No loader provided.")
            screenshot = loader()
        elif isinstance(self.screenshot, str):
            if loader is None:
                screenshot = np.asarray(Image.open(self.screenshot))
            else:
                screenshot = loader(self.screenshot)
        else:
            screenshot = self.screenshot
        return screenshot

    def __repr__(self) -> str:
        return f"UiElement(name={self.name}, bbox={self.bbox}, info={self.info})"


class UiDetectionModule:
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("A UiDetectionModule class must be callable.")
