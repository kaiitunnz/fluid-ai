import itertools
import time
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd  # type: ignore

from .base import Array, UiElement
from .icon import BaseIconLabeller
from .ocr import BaseOCR
from .ui.detection import BaseUiDetector
from .ui.filter import BaseUiFilter
from .ui.matching import BaseUiMatching


class TestUiDetectionPipeline:
    detector: BaseUiDetector
    icon_labeller: BaseIconLabeller
    icon_elements: List[str]

    def __init__(
        self,
        detector: BaseUiDetector,
        icon_labeller: BaseIconLabeller,
        icon_elements: List[str],
    ):
        self.detector = detector
        self.icon_labeller = icon_labeller
        self.icon_elements = icon_elements

    def detect(self, screenshots: Iterable[np.ndarray]) -> Iterator[List[UiElement]]:
        for detected in self.detector(list(screenshots)):
            icons = []
            for e in detected:
                if e.name in self.icon_elements:
                    icons.append(e)
            self.icon_labeller(icons)
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
            self.icon_labeller(icons)
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
    detector: BaseUiDetector
    filter: BaseUiFilter
    matcher: BaseUiMatching
    text_recognizer: BaseOCR
    textual_elements: List[str]
    icon_labeller: BaseIconLabeller
    icon_elements: List[str]

    def __init__(
        self,
        detector: BaseUiDetector,
        filter: BaseUiFilter,
        matcher: BaseUiMatching,
        text_recognizer: BaseOCR,
        textual_elements: List[str],
        icon_labeller: BaseIconLabeller,
        icon_elements: List[str],
    ):
        self.detector = detector
        self.filter = filter
        self.matcher = matcher
        self.text_recognizer = text_recognizer
        self.textual_elements = textual_elements
        self.icon_labeller = icon_labeller
        self.icon_elements = icon_elements

    def detect(
        self,
        screenshots: Iterable[Array],
        elements: Optional[Iterable[List[UiElement]]] = None,
    ) -> Iterator[List[UiElement]]:
        if elements is None:
            elements = itertools.repeat([])
        for detected, base in zip(self.detector(list(screenshots)), elements):
            filtered = self.filter(base)
            matched = self.matcher(filtered, detected)
            icons = []
            for e in matched:
                if e.name in self.icon_elements:
                    icons.append(e)
            self.icon_labeller(icons)
            yield matched

    def benchmark(
        self,
        screenshots: Iterable[Array],
        elements: Optional[Iterable[List[UiElement]]] = None,
    ) -> pd.DataFrame:
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
            self.icon_labeller(icons)
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
