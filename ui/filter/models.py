from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import torch

from .torch.datasets import RicoValidBoundary
from .torch.models import FilterModelWrapper
from ...base import Array, UiDetectionModule, UiElement, array_get_size, array_to_tensor
from ...torchutils import BatchLoader, load_model


class BaseUiFilter(UiDetectionModule):
    @abstractmethod
    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        raise NotImplementedError()

    def prefilter(self, elements: List[UiElement]) -> List[UiElement]:
        # Check bounding boxes
        elements = [
            element for element in elements if self.__class__.is_valid_bbox(element)
        ]

        return elements

    def __call__(self, elements: List[UiElement]) -> List[UiElement]:
        return self.filter(self.prefilter(elements))

    @staticmethod
    def is_valid_bbox(element: UiElement) -> bool:
        w, h = array_get_size(element.get_cropped_image())
        return w > 0 and h > 0 and element.bbox.is_valid()

    @staticmethod
    def preprocess_image(image: Array) -> torch.Tensor:
        tensor_image = torch.unsqueeze(array_to_tensor(image), 0)
        return tensor_image


class DummyUiFilter(BaseUiFilter):
    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        return elements


class TorchUiFilter(BaseUiFilter):
    model: FilterModelWrapper
    batch_size: int

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
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
        raise NotImplementedError()

    @abstractmethod
    def transform_batch(self, elements: List[UiElement]) -> List[torch.Tensor]:
        raise NotImplementedError()


class ElementUiFilter(TorchUiFilter):
    def transform(self, element: UiElement) -> torch.Tensor:
        return self.model.transform(
            self.__class__.preprocess_image(element.get_cropped_image())
        )

    def transform_batch(self, elements: List[UiElement]) -> List[torch.Tensor]:
        return [self.transform(element) for element in elements]


class BoundaryUiFilter(TorchUiFilter):
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
