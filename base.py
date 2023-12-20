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


def array_get_size(array: Array) -> Tuple[int, int]:
    if isinstance(array, np.ndarray):
        h, w = array.shape[:2]
    elif isinstance(array, torch.Tensor):
        h, w = array.size()[-2:]
    else:
        raise ValueError("Expect ndarray or Tensor. Got", type(array))
    return w, h


def array_to_tensor(array: Array) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        assert len(array.shape) == 3
        return torch.tensor(np.transpose(array, (2, 0, 1)))
    elif isinstance(array, torch.Tensor):
        return array
    raise ValueError("Expect ndarray or Tensor. Got", type(array))


def array_to_numpy(array: Array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        assert len(array.shape) == 3
        return np.array(array.permute((1, 2, 0)))
    elif isinstance(array, np.ndarray):
        return array
    raise ValueError("Expect ndarray or Tensor. Got", type(array))


class BaseBBox(NamedTuple):
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

    def is_normalized(self) -> bool:
        (x0, y0), (x1, y1) = self
        return x0 <= 1 and y0 <= 1 and x1 <= 1 and y1 <= 1


class BBox(BaseBBox):
    v0: Vertex
    v1: Vertex

    def to_int_flattened(self) -> Tuple[int, int, int, int]:
        (x0, y0), (x1, y1) = self
        return int(x0), int(y0), int(x1), int(y1)

    def normalize(self, width: Number, height: Number, check: bool = False) -> Self:
        if check:
            if self.is_normalized():
                return self
        (x0, y0), (x1, y1) = self
        return self.__class__((x0 / width, y0 / height), (x1 / width, y1 / height))

    def to_normalized(
        self,
        width: Optional[Number] = None,
        height: Optional[Number] = None,
        unchecked: bool = False,
    ) -> Optional["_NormalizedBBox"]:
        nbox = NormalizedBBox.from_bbox(self)
        if not nbox.is_valid():
            if width is None or height is None:
                return None
            nbox = NormalizedBBox.from_bbox(
                self.normalize(width, height, check=True), unchecked
            )
        return nbox

    def to_normalized_unchecked(
        self,
        width: Number,
        height: Number,
    ) -> "_NormalizedBBox":
        nbox = NormalizedBBox.from_bbox(self)
        if not nbox.is_valid():
            nbox = NormalizedBBox.from_bbox(
                self.normalize(width, height, check=True), True
            )
        return nbox

    @classmethod
    def from_normalized(cls, nbox: "_NormalizedBBox") -> Self:
        return cls(nbox.v0, nbox.v1)

    @classmethod
    def from_xywh(
        cls, xywh: Tuple[Number, Number, Number, Number], is_center: bool = False
    ):
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
        return cls((x0, y0), (x1, y1))


class NormalizedBBox:
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def new(v0: Vertex, v1: Vertex, unchecked: bool = False) -> "_NormalizedBBox":
        nbox = _NormalizedBBox(v0, v1)
        if unchecked or nbox.is_normalized():
            return nbox
        return InvalidBBox

    @classmethod
    def from_bbox(cls, bbox: BBox, unchecked: bool = False) -> "_NormalizedBBox":
        if unchecked or bbox.is_normalized():
            return _NormalizedBBox(bbox.v0, bbox.v1)
        return InvalidBBox

    @abstractmethod
    def to_bbox(self, width: Number, height: Number) -> BBox:
        raise NotImplementedError()

    def is_valid(self):
        return self is not InvalidBBox


class _NormalizedBBox(BaseBBox, NormalizedBBox):
    def __init__(self, v0: Vertex, v1: Vertex):
        super().__init__(v0, v1)

    def to_bbox(self, width: Number, height: Number) -> BBox:
        return BBox.from_normalized(self).scale(width, height)


InvalidBBox = _NormalizedBBox((0, 0), (0, 0))


class UiElement:
    name: str
    bbox: _NormalizedBBox
    screenshot: Union[str, Array, None]
    info: UiInfo

    def __init__(
        self,
        name: str,
        bbox: _NormalizedBBox,
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
        bbox = BBox.from_xywh(xywh, is_center)
        nbox = bbox.to_normalized()
        if nbox is None:
            w, h = array_get_size(screenshot)
            nbox = bbox.to_normalized(w, h)
            assert nbox is not None
        return cls(name, nbox, screenshot)

    def size(self, loader: Optional[Callable[..., Array]] = None) -> Tuple[int, int]:
        screenshot = self.get_screenshot(loader)
        w, h = array_get_size(screenshot)
        x0, y0, x1, y1 = self.bbox.to_bbox(w, h).to_int_flattened()
        return x1 - x0, y1 - y0

    def get_cropped_image(
        self, loader: Optional[Callable[..., Array]] = None
    ) -> np.ndarray:
        screenshot = self.get_screenshot(loader)
        w, h = array_get_size(screenshot)
        x0, y0, x1, y1 = self.bbox.to_bbox(w, h).to_int_flattened()
        return screenshot[y0:y1, x0:x1]

    def get_screenshot(
        self, loader: Optional[Callable[..., Array]] = None
    ) -> np.ndarray:
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
        return array_to_numpy(screenshot)

    def __repr__(self) -> str:
        return f"UiElement(name={self.name}, bbox={self.bbox}, info={self.info})"


class UiDetectionModule:
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("A UiDetectionModule class must be callable.")
