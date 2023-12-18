from abc import abstractmethod
from typing import List

import torch

from .torch.datasets import RicoValidBoundary
from ...base import Array, UiDetectionModule, UiElement
from ...torchutils import BatchLoader, ModelWrapper, load_model


class BaseUiFilter(UiDetectionModule):
    @abstractmethod
    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        raise NotImplementedError()

    def __call__(self, elements: List[UiElement]) -> List[UiElement]:
        return self.filter(elements)


class DummyUiFilter(BaseUiFilter):
    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        return elements


class ElementUiFilter(BaseUiFilter):
    model: ModelWrapper
    batch_size: int

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model = load_model(model_path).to(device)
        self.batch_size = batch_size

    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        # Assume that the screenshots are of shape (h, w, c).
        def inner(element: UiElement) -> int:
            transformed = self.model.transform(
                _preprocess_image(element.get_cropped_image())
            )
            out = self.model(transformed)
            return int(self.model.get_pred_idx(out).item())

        def inner_batched(elements: List[UiElement]) -> List[int]:
            if len(elements) == 0:
                return []
            transformed = [
                self.model.transform(_preprocess_image(element.get_cropped_image()))
                for element in elements
            ]
            result: List[int] = []
            loader = BatchLoader(self.batch_size, transformed)
            for batch in loader:
                out = self.model(batch)
                result.extend(self.model.get_pred_idx(out).to(torch.int).tolist())
            return result

        if self.batch_size == 1:
            res = [inner(element) for element in elements]
        else:
            res = inner_batched(elements)

        return [element for element, valid in zip(elements, res) if valid == 1]


class BoundaryUiFilter(BaseUiFilter):
    model: ModelWrapper
    batch_size: int

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model = load_model(model_path).to(device)
        self.batch_size = batch_size

    def filter(self, elements: List[UiElement]) -> List[UiElement]:
        # Assume that the screenshots are of shape (h, w, c).
        def inner(element: UiElement) -> int:
            transformed = self.model.transform(
                _preprocess_image(element.get_screenshot())
            )
            mask = RicoValidBoundary._get_mask(
                transformed, element.bbox, normalized=False
            )
            transformed = torch.concat([transformed, mask], dim=0)
            out = self.model(transformed)
            return int(self.model.get_pred_idx(out).item())

        def inner_batched(elements: List[UiElement]) -> List[int]:
            if len(elements) == 0:
                return []
            transformed = [
                self.model.transform(_preprocess_image(element.get_screenshot()))
                for element in elements
            ]
            transformed = [
                torch.cat(
                    [t, RicoValidBoundary._get_mask(t, e.bbox, normalized=False)], dim=0
                )
                for t, e in zip(transformed, elements)
            ]
            result: List[int] = []
            loader = BatchLoader(self.batch_size, transformed)
            for batch in loader:
                out = self.model(batch)
                result.extend(self.model.get_pred_idx(out).to(torch.int).tolist())
            return result

        if self.batch_size == 1:
            res = [inner(element) for element in elements]
        else:
            res = inner_batched(elements)

        return [element for element, valid in zip(elements, res) if valid == 1]


def _preprocess_image(image: Array) -> torch.Tensor:
    tensor_image = image if isinstance(image, torch.Tensor) else torch.tensor(image)
    tensor_image = torch.unsqueeze(torch.permute(tensor_image, (2, 0, 1)), 0)
    return tensor_image
