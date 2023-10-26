from abc import abstractmethod
from typing import Iterable, Iterator, List

import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
from ultralytics.yolo.engine.results import Results  # type: ignore

from ..base import Array, UiElement


class BaseUiDetector:
    @abstractmethod
    def detect(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        raise NotImplementedError()


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
                    ((x0, y0), (x1, y1)),
                    screenshot=screenshot,
                )
                for (x0, y0, x1, y1), cls in zip(boxes, classes)
            ]
