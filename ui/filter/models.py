from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import torch

from .torch.datasets import RicoValidBoundary
from .torch.models import FilterModelWrapper
from ...base import Array, UiDetectionModule, UiElement, array_get_size, array_to_tensor
from ...torchutils import BatchLoader, load_model


class BaseUiFilter(UiDetectionModule):
    """
    A base class for a UI filter, aka invalid UI detection model.

    A class that implements a UI filter to be used in the UI detection pipeline must
    inherit this class.
    """

    @abstractmethod
    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        """Filters out invalid UI elements

        Caller should consider using the `__call__` method instead.

        Parameters
        ----------
        elements : List[UiElement]
            List of `UiElement`s to be filtered.

        Returns
        -------
        List[UiElement]
            List of `UiElement`s with invalid UI elements filtered out.
        """
        raise NotImplementedError()

    def prefilter(self, elements: List[UiElement]) -> List[UiElement]:
        """Filters out invalid UI elements based on heuristics

        It can be used to preprocess the input list of UI elements to ensure that
        the actual filtering algorithm works properly. The default implementation
        is to filter out elements with invalid bounding boxes.

        Parameters
        ----------
        elements : List[UiElement]
            List of `UiElement`s to be preprocessed.

        Returns
        -------
        List[UiElement]
            List of preprocessed `UiElement`s with some `UIElement`s filtered out
            based on a set of rules.
        """
        # Check bounding boxes
        elements = [
            element for element in elements if self.__class__.is_valid_bbox(element)
        ]

        return elements

    def __call__(self, elements: List[UiElement]) -> List[UiElement]:
        """Filters out invalid UI elements

        It calls `self.prefilter` followed by `self.filter`.

        Parameters
        ----------
        elements : List[UiElement]
            List of `UiElement`s to be filtered.

        Returns
        -------
        List[UiElement]
            List of `UiElement`s with invalid UI elements filtered out.
        """
        return self.filter(self.prefilter(elements))

    @staticmethod
    def is_valid_bbox(element: UiElement) -> bool:
        """Tests if a `UiElement`'s bounding box is valid

        Parameters
        ----------
        element : UiElement
            A `UiElement` to be tested.

        Returns
        -------
        bool
            Wheter a `UiElement` is valid.
        """
        w, h = array_get_size(element.get_cropped_image())
        return w > 0 and h > 0 and element.bbox.is_valid()

    @staticmethod
    def preprocess_image(image: Array) -> torch.Tensor:
        """Preprocesses an image for model's inference

        Parameters
        ----------
        image : Array
            An image to be preprocessed. Must conform to the Array format.

        Returns
        -------
        Tensor
            The resulting PyTorch Tensor.
        """
        tensor_image = array_to_tensor(image)
        return tensor_image


class DummyUiFilter(BaseUiFilter):
    """
    A dummy UI filter.

    Simply returns the input.
    """

    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        return elements


class TorchUiFilter(BaseUiFilter):
    """
    A base class of UI filters based on PyTorch models.

    Attributes
    ----------
    model : FilterModelWrapper
        A UI filter model wrapped in a `FilterModelWrapper`.
    batch_size : int
        Batch size for batched inference.
    """

    model: FilterModelWrapper
    batch_size: int

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to the model checkpoint. It should have been saved by
            `fluid_ai.torchutils.save_model`.
        batch_size : int
            Batch size for batched inference.
        device : device
            The device on which the model will run.
        """
        super().__init__()
        model = load_model(model_path).to(device)
        assert isinstance(model, FilterModelWrapper)
        self.model = model
        self.batch_size = batch_size

    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        # Assume that the screenshots are of shape (h, w, c).
        def infer(element: UiElement) -> int:
            out = self.model(self.transform(element))
            return int(self.model.get_pred_idx(out).item())

        def infer_batched(elements: List[UiElement]) -> List[int]:
            if len(elements) == 0:
                return []
            transformed = self.transform_batch(elements)
            result: List[int] = []
            loader = BatchLoader(self.batch_size, transformed)
            for batch in loader:
                out = self.model(batch)
                result.extend(
                    self.model.get_pred_idx(out).view((-1,)).to(torch.int).tolist()
                )
            return result

        if self.batch_size == 1:
            res = [infer(element) for element in elements]
        else:
            res = infer_batched(elements)

        return [element for element, valid in zip(elements, res) if valid == 1]

    @abstractmethod
    def transform(self, element: UiElement) -> torch.Tensor:
        """Transforms a `UiElement` into a Tensor, which will be input to the
        PyTorch model.

        Parameters
        ----------
        element : UiElement
            A UI element to be transformed.

        Returns
        -------
        Tensor
            The resulting Tensor.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform_batch(self, elements: List[UiElement]) -> List[torch.Tensor]:
        """Transforms `UiElement`s in a list, which will be input to the PyTorch model.

        Parameters
        ----------
        elements : List[UiElement]
            List of `UiElement`s to be transformed.

        Returns
        -------
        List[Tensor]
            List of resulting Tensors.
        """
        raise NotImplementedError()


class ElementUiFilter(TorchUiFilter):
    """
    An element-based UI filter.
    """

    def transform(self, element: UiElement) -> torch.Tensor:
        return self.model.transform(
            self.__class__.preprocess_image(element.get_cropped_image())
        )

    def transform_batch(self, elements: List[UiElement]) -> List[torch.Tensor]:
        return [self.transform(element) for element in elements]


class BoundaryUiFilter(TorchUiFilter):
    """
    A boundary-based UI filter.
    """

    def transform(self, element: UiElement) -> torch.Tensor:
        transformed = self.model.transform(
            self.__class__.preprocess_image(element.get_screenshot())
        )
        return RicoValidBoundary.mask(transformed, element.bbox)

    def transform_batch(self, elements: List[UiElement]) -> List[torch.Tensor]:
        if len(elements) == 0:
            return []
        screenshot: torch.Tensor = self.model.transform(
            self.__class__.preprocess_image(elements[0].get_screenshot())
        )
        return [
            RicoValidBoundary.mask(screenshot, element.bbox) for element in elements
        ]
