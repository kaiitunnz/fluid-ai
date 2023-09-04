from abc import abstractmethod
from typing import Iterable, Iterator, List

import numpy as np
from ultralytics import YOLO  # type: ignore
from ultralytics.yolo.engine.results import Results  # type: ignore

from ..base import UiElement


class BaseUiDetector:
    @abstractmethod
    def detect(self, screenshots: Iterable[np.ndarray]) -> Iterator[List[UiElement]]:
        raise NotImplementedError()


class YoloUiDetector(BaseUiDetector):
    model: YOLO

    def __init__(self, model_path: str):
        self.model = YOLO(model_path, task="detect")

    def detect(self, screenshots: Iterable[np.ndarray]) -> Iterator[List[UiElement]]:
        result_list: List[Results] = self.model.predict(screenshots, verbose=False)
        for results in result_list:
            if results.boxes is None:
                continue
            boxes, classes = (
                results.boxes.data[:, :4].tolist(),
                results.boxes.data[:, 5].tolist(),
            )
            yield [
                UiElement(
                    results.names[int(cls)],
                    ((x0, y0), (x1, y1)),
                    screenshot=results.orig_img,
                )
                for (x0, y0, x1, y1), cls in zip(boxes, classes)
            ]
