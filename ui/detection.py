from abc import abstractmethod
from typing import Iterable, Iterator, List

import numpy as np
import torch
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Results  # type: ignore

from ..base import (
    Array,
    BBox,
    NormalizedBBox,
    UiDetectionModule,
    UiElement,
    array_get_size,
)


class BaseUiDetector(UiDetectionModule):
    """
    A base class for UI element detection.

    A class that implements a UI detector to be used in the UI detection pipeline must
    inherit this class.
    """

    @abstractmethod
    def detect(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        """Detects UI elements in screenshots

        A UI detector must implement this method.

        Parameters
        ----------
        screenshots : Iterable[Array]
            An iterable object of screenshots. The screenshots must be of type either
            a NumPy array of shape H x W x C or a PyTorch tensor of shape C x H x W.
        save_img : bool
            Whether to store the original screenshot in the returned UI elements. If true,
            the UiElement's screenshot field will refer to the original screenshot array.

        Returns
        -------
        Iterator[List[UiElement]]
            An iterator of lists of UI elements detected in the screenshots.
        """
        raise NotImplementedError()

    def __call__(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        """Detects UI elements in screenshots

        Parameters
        ----------
        screenshots : Iterable[Array]
            An iterable object of screenshots. The screenshots must be of type either
            a NumPy array of shape H x W x C or a PyTorch tensor of shape C x H x W.
        save_img : bool
            Whether to store the original screenshot in the returned UI elements. If true,
            the UiElement's screenshot field will refer to the original screenshot array.

        Returns
        -------
        Iterator[List[UiElement]]
            An iterator of lists of UI elements detected in the screenshots.
        """
        return self.detect(screenshots, save_img)


class DummyUiDetector(BaseUiDetector):
    """
    A dummy UI element detector.

    It returns a single dummy UI element of class SCREEN per screenshot, representing
    the entire screen.
    """

    @abstractmethod
    def detect(
        self, screenshots: Iterable[Array], save_img: bool = True
    ) -> Iterator[List[UiElement]]:
        for screenshot in screenshots:
            yield [UiElement("SCREEN", NormalizedBBox.new((0, 0), (1, 1)), screenshot)]


class YoloUiDetector(BaseUiDetector):
    """
    A UI detector based on Uiltralytics' YOLOv8.

    Attributes
    ----------
    model : YOLO
        The internal YOLO object-detection model.
    device : torch.device
        The device on which the YOLO model is running. Due to YOLOv8's restriction,
        it must always be set to `cuda:0`.
    """

    model: YOLO
    device: torch.device

    def __init__(self, model_path: str, device: torch.device = torch.device("cpu")):
        """
        Parameters
        ----------
        model_path : str
            Path to the model checkpoint, which should be trained by Ultralytics'
            training utilities.
        device : torch.device
            The device on which the YOLO model is running. Due to YOLOv8's restriction,
            it must always be set to `cuda:0`.
        """
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
