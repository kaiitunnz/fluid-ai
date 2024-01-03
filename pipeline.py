import itertools
import time
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd  # type: ignore

from .base import Array, UiElement
from .icon import BaseIconLabeler
from .ocr import BaseOCR
from .ui.detection import BaseUiDetector
from .ui.filter import BaseUiFilter
from .ui.matching import BaseUiMatching


class TestUiDetectionPipeline:
    """
    A UI detection pipeline for testing specific UI detection modules.

    It should not be used in a deployment environment.
    """

    detector: BaseUiDetector
    icon_labeler: BaseIconLabeler
    icon_elements: List[str]

    def __init__(
        self,
        detector: BaseUiDetector,
        icon_labeler: BaseIconLabeler,
        icon_elements: List[str],
    ):
        self.detector = detector
        self.icon_labeler = icon_labeler
        self.icon_elements = icon_elements

    def detect(self, screenshots: Iterable[np.ndarray]) -> Iterator[List[UiElement]]:
        for detected in self.detector(list(screenshots)):
            icons = []
            for e in detected:
                if e.name in self.icon_elements:
                    icons.append(e)
            self.icon_labeler(icons)
            yield detected

    def benchmark(self, screenshots: Iterable[np.ndarray]) -> pd.DataFrame:
        icon_elements = set(self.icon_elements)
        columns = [
            "num_detected_total",
            "num_detected_icons",
            "ui_detection_time (ms)",
            "icon_labelling_time (ms)",
        ]
        results = []
        first = True
        for img in screenshots:
            if first:
                _ = next(self.detect([img]))  # warm up
                first = False
            start = time.time()
            detected: List[UiElement] = next(self.detector([img]))
            ui_detection_time = time.time() - start  # 2
            num_detected = len(detected)  # 0
            icons = []
            for e in detected:
                if e.name in icon_elements:
                    icons.append(e)
            num_detected_icons = len(icons)  # 1
            start = time.time()
            self.icon_labeler(icons)
            icon_labelling_time = time.time() - start  # 3
            results.append(
                [
                    num_detected,
                    num_detected_icons,
                    ui_detection_time * 1000,
                    icon_labelling_time * 1000,
                ]
            )
        return pd.DataFrame(results, columns=columns)


class UiDetectionPipeline:
    """
    A sequential UI detection pipeline.

    Attributes
    ----------
    detector : BaseUiDetector
        UI detection model.
    filter : BaseUiFilter
        UI filter model, aka invalid UI detection model.
    matcher : BaseUiMatching
        UI matching model.
    text_recognizer : BaseOCR
        Text recognition module.
    textual_elements : List[str]
        Names of UI classes corresponding to textual UI elements.
    icon_labeler : BaseIconLabeler
        Icon labeling module.
    icon_elements : List[str]
        Names of UI classes corresponding to icon UI elements.
    """

    detector: BaseUiDetector
    filter: BaseUiFilter
    matcher: BaseUiMatching
    text_recognizer: BaseOCR
    textual_elements: List[str]
    icon_labeler: BaseIconLabeler
    icon_elements: List[str]

    def __init__(
        self,
        detector: BaseUiDetector,
        filter: BaseUiFilter,
        matcher: BaseUiMatching,
        text_recognizer: BaseOCR,
        textual_elements: List[str],
        icon_labeler: BaseIconLabeler,
        icon_elements: List[str],
    ):
        """
        Parameters
        ----------
        detector : BaseUiDetector
            UI detection model.
        filter : BaseUiFilter
            UI filter model, aka invalid UI detection model.
        matcher : BaseUiMatching
            UI matching model.
        text_recognizer : BaseOCR
            Text recognition module.
        textual_elements : List[str]
            Names of UI classes corresponding to textual UI elements.
        icon_labeler : BaseIconLabeler
            Icon labeling module.
        icon_elements : List[str]
            Names of UI classes corresponding to icon UI elements.
        """
        self.detector = detector
        self.filter = filter
        self.matcher = matcher
        self.text_recognizer = text_recognizer
        self.textual_elements = textual_elements
        self.icon_labeler = icon_labeler
        self.icon_elements = icon_elements

    def detect(
        self,
        screenshots: Iterable[Array],
        elements: Optional[Iterable[List[UiElement]]] = None,
    ) -> Iterator[List[UiElement]]:
        """Detects UI elements and extract their auxiliary information from the given
        screenshots

        Parameters
        ----------
        screenshots : Iterable[Array]
            Screenshots to detect UI elements.
        elements : Optional[Iterable[List[UiElement]]]
            Lists of additional UI elements.

        Returns
        -------
        Iterator[List[UiElement]]
            Lists of detected UI elements.
        """
        if elements is None:
            elements = itertools.repeat([])
        for detected, base in zip(self.detector(list(screenshots)), elements):
            filtered = self.filter(base)
            matched = self.matcher(filtered, detected)
            icons = []
            for e in matched:
                if e.name in self.icon_elements:
                    icons.append(e)
            self.icon_labeler(icons)
            yield matched

    def benchmark(
        self,
        screenshots: Iterable[Array],
        elements: Optional[Iterable[List[UiElement]]] = None,
    ) -> pd.DataFrame:
        """Benchmarks the UI detection pipeline using the given screenshots.

        Parameters
        ----------
        screenshots : Iterable[Array]
            Screenshots to detect UI elements.
        elements : Optional[Iterable[List[UiElement]]]
            Lists of additional UI elements.

        Returns
        -------
        DataFrame
            Benchmark results.
        """
        if elements is None:
            elements = itertools.repeat([])
        textual_elements = set(self.textual_elements)
        icon_elements = set(self.icon_elements)
        columns = [
            "num_detected_total",
            "num_matched",
            "num_detected_textual",
            "num_detected_icons",
            "ui_detection_time (ms)",
            "invalid_ui_detection_time (ms)",
            "matching_time (ms)",
            "text_recognition_time (ms)",
            "icon_labelling_time (ms)",
        ]
        results = []
        first = True
        for img, base in zip(screenshots, elements):
            if first:
                _ = next(self.detect([img]))  # warm up
                first = False
            start = time.time()
            detected: List[UiElement] = next(self.detector([img]))
            ui_detection_time = time.time() - start  # 4
            num_detected = len(detected)  # 0
            start = time.time()
            filtered = self.filter(base)
            filter_time = time.time() - start  # 5
            num_filtered = len(filtered)
            start = time.time()
            matched = self.matcher(filtered, detected)
            matching_time = time.time() - start  # 6
            num_matched = len(matched)  # 1
            textual = []
            icons = []
            for e in matched:
                if e.name in textual_elements:
                    textual.append(e)
                if e.name in icon_elements:
                    icons.append(e)
            num_matched_textual = len(textual)  # 2
            num_matched_icons = len(icons)  # 3
            start = time.time()
            self.text_recognizer(textual)
            text_recognition_time = time.time() - start  # 7
            start = time.time()
            self.icon_labeler(icons)
            icon_labelling_time = time.time() - start  # 8
            results.append(
                [
                    num_detected,
                    num_filtered,
                    num_matched,
                    num_matched_textual,
                    num_matched_icons,
                    ui_detection_time * 1000,
                    filter_time * 1000,
                    matching_time * 1000,
                    text_recognition_time * 1000,
                    icon_labelling_time * 1000,
                ]
            )
        return pd.DataFrame(results, columns=columns)
