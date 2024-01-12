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
UiRelation = Dict[str, Any]


def array_get_size(array: Array) -> Tuple[int, int]:
    """Gets the size (width, height) of the array

    Parameters
    ----------
    array : Array
        An array that conforms to the Array shape.

    Returns
    -------
    Tuple[int, int]
        width, height of the array.
    """
    if isinstance(array, np.ndarray):
        h, w = array.shape[:2]
    elif isinstance(array, torch.Tensor):
        h, w = array.size()[-2:]
    else:
        raise ValueError("Expect ndarray or Tensor. Got", type(array))
    return w, h


def array_to_tensor(array: Array) -> torch.Tensor:
    """Converts the given array to a PyTorch Tensor

    Parameters
    ----------
    array : Array
        An array that conforms to the Array shape.

    Returns
    -------
    Tensor
        Tensor representation of the array.
    """
    if isinstance(array, np.ndarray):
        assert len(array.shape) == 3
        return torch.tensor(np.transpose(array, (2, 0, 1)))
    elif isinstance(array, torch.Tensor):
        return array
    raise ValueError("Expect ndarray or Tensor. Got", type(array))


def array_to_numpy(array: Array) -> np.ndarray:
    """Converts the given array to a NumPy ndarray

    Parameters
    ----------
    array : Array
        An array that conforms to the Array shape.

    Returns
    -------
    ndarray
        NumPy ndarray representation of the array.
    """
    if isinstance(array, torch.Tensor):
        assert len(array.shape) == 3
        return np.array(array.permute((1, 2, 0)))
    elif isinstance(array, np.ndarray):
        return array
    raise ValueError("Expect ndarray or Tensor. Got", type(array))


class BaseBBox(NamedTuple):
    """
    A base class for a bounding box.

    Attributes
    ----------
    v0 : Vertex
        The top-left vertex of the bounding box.
    v1 : Vertex
        The bottom-right vertex of the bounding box.
    """

    v0: Vertex
    v1: Vertex

    @classmethod
    def new(cls, x0: Number, y0: Number, x1: Number, y1: Number) -> Self:
        """Creates a new instance

        Parameters
        ----------
        x0 : Number
            Left-most x-coordinate.
        y0 : Number
            Top-most y-coordinate.
        x1 : Number
            Right-most x-coordinate.
        y1 : Number
            Bottom-most y-coordinate.

        Returns
        -------
        BaseBBox
            A new instance.
        """
        return cls((x0, y0), (x1, y1))

    def flatten(self) -> Tuple[Number, Number, Number, Number]:
        """Returns a flattened tuple representation

        Returns
        -------
        Tuple[Number, Number, Number, Number]
            Flattened tuple representation of the bounding box.
        """
        return *self.v0, *self.v1

    def scale(self, scale_x: Number, scale_y: Number) -> Self:
        """Creates a scaled bounding box

        Parameters
        ----------
        scale_x : Number
            Scale along the x-axis.
        scale_y : Number
            Scale along the y-axis.

        Returns
        -------
        BaseBBox
            A new instance with the scaled size.
        """
        (x0, y0), (x1, y1) = self
        return self.__class__(
            (x0 * scale_x, y0 * scale_y), (x1 * scale_x, y1 * scale_y)
        )

    def map(self, func: Callable[[Number], Number]) -> Self:
        """Creates a new bounding box with `func` applied to each coordinate of
        each vertex

        Parameters
        ----------
        func : Callable[[Number], Number]
            A function to be applied to each coordinate of each vertex.

        Returns
        -------
        BaseBBox
            A new instance with `func` applied to each coordinate of each vertex.
        """
        (x0, y0), (x1, y1) = self
        return self.__class__((func(x0), func(y0)), (func(x1), func(y1)))

    def is_normalized(self) -> bool:
        """Returns whether the bounding box is normalized according to a heuristic

        Returns
        -------
        bool
            Whether the bounding box is normalized.
        """
        (x0, y0), (x1, y1) = self
        return x0 <= 1 and y0 <= 1 and x1 <= 1 and y1 <= 1


class BBox(BaseBBox):
    """
    A general bounding box.

    Attributes
    ----------
    v0 : Vertex
        The top-left vertex of the bounding box.
    v1 : Vertex
        The bottom-right vertex of the bounding box.
    """

    def to_int_flattened(self) -> Tuple[int, int, int, int]:
        """Returns a flatted tuple representation of the bounding box which each
        coordinate converted to int

        Returns
        -------
        Tuple[int, int, int, int]
            Flatted tuple representation of the bounding box which each coordinate
            converted to int.
        """
        (x0, y0), (x1, y1) = self
        return int(x0), int(y0), int(x1), int(y1)

    def normalize(self, width: Number, height: Number, check: bool = False) -> Self:
        """Normalizes the size of the bounding box

        Parameters
        ----------
        width : Number
            Total width of the image containing the bounding box.
        height : Number
            Total height of the image containing the bounding box.
        check : bool
            Whether to check the bounding box before normalizing. If true, return
            itself if the bounding box is already normalized. Otherwise, return a
            normalized bounding box regardless of whether it is already normalized.

        Returns
        -------
        BBox
            A bounding box with normalized coordinates.
        """
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
        """Converts the bounding box to a normalized bounding box according to `width`
        and `height`

        If the bounding box is already normalized and valid, return a `NormalizedBBox`
        with the original size immediately. Otherwise, normalize the bounding
        box with `width` and `height`. If either of them is `None`, return `None`.

        If `unchecked`, the bounding box will be normalized at most once, and the
        resulting bounding box can have the width or height greater than one. Otherwise,
        if the resulting bounding box is not totally normalized, return `InvalidBBox`.

        Parameters
        ----------
        width : Optional[Number]
            Total width of the image containing the bounding box. `None` if the bounding
            box is already normalized.
        height : Optional[Number]
            Total height of the image containing the bounding box. `None` if the bounding
            box is already normalized.
        unchecked: bool
            Whether to check the resulting normalized bounding box.

        Returns
        -------
        NormalizedBBox
            The resulting normalized bounding box. `None` if the bounding box is not
            already normalized and the width or height is `None`.
        """
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
        """Converts the bounding box to a normalized bounding box according to `width`
        and `height` without checking the resulting bounding box

        If the bounding box is already normalized and valid, return a `NormalizedBBox`
        with the original size immediately. Otherwise, normalize the bounding box
        with `width` and `height` and return the resulting normalized bounding box.

        Parameters
        ----------
        width : Optional[Number]
            Total width of the image containing the bounding box.
        height : Optional[Number]
            Total height of the image containing the bounding box.

        Returns
        -------
        NormalizedBBox
            The resulting normalized bounding box.
        """
        nbox = NormalizedBBox.from_bbox(self)
        if not nbox.is_valid():
            nbox = NormalizedBBox.from_bbox(
                self.normalize(width, height, check=True), True
            )
        return nbox

    @classmethod
    def from_normalized(cls, nbox: "_NormalizedBBox") -> Self:
        """Create a `BBox` from a `NormalizedBBox` without scaling

        Parameters
        ---------
        nbox : NormalizedBBox
            A normalized bounding box from which the `BBox` is to be created.

        Returns
        -------
        BBox
            The resulting bounding box.
        """
        return cls(nbox.v0, nbox.v1)

    @classmethod
    def from_xywh(
        cls, xywh: Tuple[Number, Number, Number, Number], is_center: bool = False
    ) -> Self:
        """Create a `BBox` from the anchor point (x, y) and size (w, h)

        If `is_center`, the center point is assumed to be the anchor point of the
        bounding box. Otherwise, the top-left corner is used.

        Parameters
        ----------
        xywh : Tuple[Number, Number, Number, Number]
            (x, y, w, h).
        is_center : bool
            Whether the anchor point (x, y) is the center.

        Returns
        -------
        BBox
            The resulting bounding box.
        """
        x, y, w, h = xywh
        if is_center:
            x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        else:
            x0, y0 = x, y
            x1, y1 = x + w, y + h
        return cls((x0, y0), (x1, y1))


class NormalizedBBox:
    """
    A public interface representing a normalized bounding box.

    It cannot be instantiated.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def new(v0: Vertex, v1: Vertex, unchecked: bool = False) -> "_NormalizedBBox":
        """Creates a normalized bounding box

        If `unchecked`, always return the resulting bounding box without checking
        the validity. Otherwise, return `InvalidBBox` if the coordinates are not
        normalized.

        Parameters
        ----------
        v0 : Vertex
            Normalized coordinate of the top-left corner.
        v1 : Vertex
            Normalized coordinate of the bottom-right corner.
        unchecked : bool
            Whether to check the resulting bounding box.

        Returns
        -------
        NormalizedBBox
            The resulting normalized bounding box.
        """
        nbox = _NormalizedBBox(v0, v1)
        if unchecked or nbox.is_normalized():
            return nbox
        return InvalidBBox

    @staticmethod
    def from_bbox(bbox: BBox, unchecked: bool = False) -> "_NormalizedBBox":
        """Creates a normalized bounding box from a `BBox`

        If `unchecked`, always return the resulting bounding box without checking
        the validity. Otherwise, return `InvalidBBox` if the coordinates are not
        normalized.

        Parameters
        ----------
        bbox : BBox
            The `BBox` from which the `NormalizedBBox` is to be created.
        unchecked : bool
            Whether to check the resulting bounding box.

        Returns
        -------
        NormalizedBBox
            The resulting normalized bounding box.
        """
        if unchecked or bbox.is_normalized():
            return _NormalizedBBox(bbox.v0, bbox.v1)
        return InvalidBBox

    @abstractmethod
    def to_bbox(self, width: Number, height: Number) -> BBox:
        """Creates a `BBox` from a `NormalizedBBox` given the `width` and `height`
        of the original image containing the bounding box

        The normalized bounding box will be scaled according to `width` and `height`.

        Parameters
        ----------
        width : Number
            Total width of the image containing the bounding box.
        height : Number
            Total height of the image containing the bounding box.

        Returns
        -------
        BBox
            The resulting bounding box.
        """
        raise NotImplementedError()

    def is_valid(self) -> bool:
        """Returns whether the normalized bounding box is valid

        Check if the normalized bounding box is the `InvalidBBox`.

        Returns
        -------
        bool
            Whether the normalized bounding box is valid.
        """
        return self is not InvalidBBox


class _NormalizedBBox(BaseBBox, NormalizedBBox):
    """
    The internal class representing a normalized bounding box.

    It should only be instantiated through `NormalizedBBox`.

    Attributes
    ----------
    v0 : Vertex
        The top-left vertex of the bounding box.
    v1 : Vertex
        The bottom-right vertex of the bounding box.
    """

    def __init__(self, v0: Vertex, v1: Vertex):
        """
        Parameters
        ----------
        v0 : Vertex
            The top-left vertex of the bounding box.
        v1 : Vertex
            The bottom-right vertex of the bounding box.
        """
        super().__init__(v0, v1)

    def to_bbox(self, width: Number, height: Number) -> BBox:
        return BBox.from_normalized(self).scale(width, height)


# Object representing an invalid normalized bounding box
InvalidBBox = _NormalizedBBox((0, 0), (0, 0))


class UiElement:
    """
    A class representing a UI element.

    Attributes
    ----------
    name : str
        Name of the UI element. It can represent the UI class.
    bbox : NormalizedBBox
        Bounding box of the UI element.
    screenshot : Union[str, Array, None]
        The screenshot containing the UI element. A `str` should be a pointer to the
        screenshot that can be utilized by a `loader` (see `UiElement.size()`, `UiElement.get_cropped_image()`
        and `UiElement.get_screenshot()`).
    info : UiInfo
        Auxiliary information of the UI element.
    relation : UiRelation
        Relations between UI elements.
    """

    name: str
    bbox: _NormalizedBBox
    screenshot: Union[str, Array, None]
    info: UiInfo
    relation: UiRelation

    def __init__(
        self,
        name: str,
        bbox: _NormalizedBBox,
        screenshot: Optional[Union[str, Array]] = None,
        info: Optional[UiInfo] = None,
        relation: Optional[UiRelation] = None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the UI element. It can represent the UI class.
        bbox : NormalizedBBox
            Bounding box of the UI element.
        screenshot : Union[str, Array, None]
            The screenshot containing the UI element. A `str` should be a pointer
            to the screenshot that can be utilized by a `loader` (see `UiElement.size()`,
            `UiElement.get_cropped_image()` and `UiElement.get_screenshot()`).
        info : UiInfo
            Auxiliary information of the UI element.
        """
        self.name = name
        self.bbox = bbox
        self.screenshot = screenshot
        self.info = {} if info is None else info
        self.relation = {} if relation is None else relation

    @classmethod
    def from_xywh(
        cls,
        name: str,
        xywh: Tuple[Number, Number, Number, Number],
        screenshot: Array,
        is_center: bool = False,
    ) -> Self:
        """Creates a `UiElement` with a bounding box from the anchor point (x, y)
        and size (w, h)

        Parameters
        ----------
        name : str
            Name of the UI element. It can represent the UI class.
        xywh : Tuple[Number, Number, Number, Number]
            (x, y, w, h).
        screenshot : Array
            The screenshot containing the UI element.
        is_center : bool
            Whether the anchor point (x, y) is the center.

        Returns
        -------
        UiElement
            The resulting UI element.
        """
        bbox = BBox.from_xywh(xywh, is_center)
        nbox = bbox.to_normalized()
        if nbox is None:
            w, h = array_get_size(screenshot)
            nbox = bbox.to_normalized(w, h)
            assert nbox is not None
        return cls(name, nbox, screenshot)

    def size(self, loader: Optional[Callable[..., Array]] = None) -> Tuple[int, int]:
        """Returns the unnormalized size (width, height) of the UI element

        Parameters
        ----------
        loader : Optional[Callable[..., Array]]
            Loader to load the screenshot. None if the screenshot is already loaded
            or the default loader is to be used.

        Returns
        -------
        Tuple[int, int]
            Unnormalized (width, height) of the UI element.
        """
        screenshot = self.get_screenshot(loader)
        w, h = array_get_size(screenshot)
        x0, y0, x1, y1 = self.bbox.to_bbox(w, h).to_int_flattened()
        return x1 - x0, y1 - y0

    def get_cropped_image(
        self, loader: Optional[Callable[..., Array]] = None
    ) -> np.ndarray:
        """Returns the image of the UI element cropped from the screenshot

        Parameters
        ----------
        loader : Optional[Callable[..., Array]]
            Loader to load the screenshot. None if the screenshot is already loaded
            or the default loader is to be used.

        Returns
        -------
        ndarray
            Image of the UI element cropped from the screenshot.
        """
        screenshot = self.get_screenshot(loader)
        w, h = array_get_size(screenshot)
        x0, y0, x1, y1 = self.bbox.to_bbox(w, h).to_int_flattened()
        return screenshot[y0:y1, x0:x1]

    def get_screenshot(
        self, loader: Optional[Callable[..., Array]] = None
    ) -> np.ndarray:
        """Returns the screenshot containing the UI element

        Parameters
        ----------
        loader : Optional[Callable[..., Array]]
            Loader to load the screenshot. None if the screenshot is already loaded
            or the default loader is to be used.

        Returns
        -------
        ndarray
            Screenshot containing the UI element.
        """
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
        return f"UiElement(name={self.name}, bbox={self.bbox}, info={self.info}, relation={self.relation})"


class UiDetectionModule:
    """
    A parent class of all UI detection modules.

    It provides a common interface for calling the modules.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("A UiDetectionModule class must be callable.")
