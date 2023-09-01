import time
from typing import Iterable, List

import numpy as np
import pandas as pd

from .base import UiElement
from .icon import BaseIconLabeller
from .ocr import BaseOCR
from .ui.detection import BaseUiDetector


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

    def detect(self, screenshots: Iterable[np.ndarray]) -> Iterable[List[UiElement]]:
        for detected in self.detector.detect(list(screenshots)):
            icons = []
            for e in detected:
                if e.name in self.icon_elements:
                    icons.append(e)
            self.icon_labeller.process(icons)
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
            detected: List[UiElement] = next(self.detector.detect([img]))
            ui_detection_time = time.time() - start  # 2
            num_detected = len(detected)  # 0
            icons = []
            for e in detected:
                if e.name in icon_elements:
                    icons.append(e)
            num_detected_icons = len(icons)  # 1
            start = time.time()
            self.icon_labeller.process(icons)
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
    text_recognizer: BaseOCR
    textual_elements: List[str]
    icon_labeller: BaseIconLabeller
    icon_elements: List[str]

    def __init__(
        self,
        detector: BaseUiDetector,
        text_recognizer: BaseOCR,
        textual_elements: List[str],
        icon_labeller: BaseIconLabeller,
        icon_elements: List[str],
    ):
        self.detector = detector
        self.text_recognizer = text_recognizer
        self.textual_elements = textual_elements
        self.icon_labeller = icon_labeller
        self.icon_elements = icon_elements

    def detect(self, screenshots: Iterable[np.ndarray]) -> Iterable[List[UiElement]]:
        for detected in self.detector.detect(screenshots):
            textual = []
            icons = []
            for e in detected:
                if e.name in self.textual_elements:
                    textual.append(e)
                if e.name in self.icon_elements:
                    icons.append(e)
            self.text_recognizer.process(textual)
            self.icon_labeller.process(icons)
            yield detected

    def benchmark(self, screenshots: Iterable[np.ndarray]) -> pd.DataFrame:
        textual_elements = set(self.textual_elements)
        icon_elements = set(self.icon_elements)
        columns = [
            "num_detected_total",
            "num_detected_textual",
            "num_detected_icons",
            "ui_detection_time (ms)",
            "text_recognition_time (ms)",
            "icon_labelling_time (ms)",
        ]
        results = []
        first = True
        for img in screenshots:
            if first:
                _ = next(self.detect([img]))  # warm up
                first = False
            start = time.time()
            detected: List[UiElement] = next(self.detector.detect([img]))
            ui_detection_time = time.time() - start  # 3
            num_detected = len(detected)  # 0
            textual = []
            icons = []
            for e in detected:
                if e.name in textual_elements:
                    textual.append(e)
                if e.name in icon_elements:
                    icons.append(e)
            num_detected_textual = len(textual)  # 1
            num_detected_icons = len(icons)  # 2
            start = time.time()
            self.text_recognizer.process(textual)
            text_recognition_time = time.time() - start  # 4
            start = time.time()
            self.icon_labeller.process(icons)
            icon_labelling_time = time.time() - start  # 5
            results.append(
                [
                    num_detected,
                    num_detected_textual,
                    num_detected_icons,
                    ui_detection_time * 1000,
                    text_recognition_time * 1000,
                    icon_labelling_time * 1000,
                ]
            )
        return pd.DataFrame(results, columns=columns)
