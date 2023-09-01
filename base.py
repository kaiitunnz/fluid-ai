from typing import Any, Dict, Tuple, Union
from typing_extensions import Self

import numpy as np

Number = Union[int, float]
Vertex = Tuple[Number, Number]
Box = Tuple[Vertex, Vertex, Vertex, Vertex]
BBox = Tuple[Vertex, Vertex]
UiInfo = Dict[str, Any]


class UiElement:
    name: str
    bbox: BBox
    screenshot: np.ndarray
    info: UiInfo

    def __init__(self, name: str, bbox: BBox, screenshot: np.ndarray):
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
        self.info = {}

    @classmethod
    def from_xywh(
        cls,
        name: str,
        xywh: Tuple[Number, Number, Number, Number],
        screenshot: np.ndarray,
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
        return cls(name, ((x0, y0), (x1, y1)), screenshot)

    def size(self) -> Tuple[Number, Number]:
        (x0, y0), (x1, y1) = self.bbox
        return abs(x0 - x1), abs(y0 - y1)

    def get_cropped_image(self) -> np.ndarray:
        (x0, y0), (x1, y1) = self.bbox
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        return self.screenshot[y0:y1, x0:x1]

    def __repr__(self) -> str:
        return f"UiElement(name={self.name}, bbox={self.bbox}, info={self.info})"
