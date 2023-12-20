from abc import abstractmethod
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Results  # type: ignore

from ..base import (
    Array,
    BBox,
    NormalizedBBox,
    Number,
    UiDetectionModule,
    UiElement,
    array_get_size,
)


class BaseUiDetector(UiDetectionModule):
    @abstractmethod
    def detect(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        raise NotImplementedError()

    def __call__(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        return self.detect(screenshots, save_img)


class DummyUiDetector(BaseUiDetector):
    @abstractmethod
    def detect(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        for screenshot in screenshots:
            shape: Tuple[Number, Number] = tuple(screenshot.shape[:2])  # type: ignore
            yield [UiElement("SCREEN", NormalizedBBox.new((0, 0), (1, 1)), screenshot)]


class YoloUiDetector(BaseUiDetector):
    model: YOLO
    device: torch.device

    def __init__(self, model_path: str, device: torch.device = torch.device("cpu")):
        self.model = YOLO(model_path, task="detect")
        self.model.to(device)
        self.device = device

    def detect(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        screenshots = [np.asarray(screenshot) for screenshot in screenshots]
        result_list: List[Results] = self.model.predict(
            screenshots, verbose=False, device=self.device
        )
        for results in result_list:
            if results.boxes is None:
                continue
            boxes, classes = (
                results.boxes.data[:, :4].tolist(),
                results.boxes.data[:, 5].tolist(),
            )
            screenshot = results.orig_img if save_img else None
            yield [
                UiElement(
                    results.names[int(cls)],
                    BBox((x0, y0), (x1, y1)).to_normalized_unchecked(
                        *array_get_size(results.orig_img)
                    ),
                    screenshot=screenshot,
                )
                for (x0, y0, x1, y1), cls in zip(boxes, classes)
            ]
