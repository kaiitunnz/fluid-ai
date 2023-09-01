from abc import abstractmethod
from typing import Any, Iterable, List, Tuple, Union

import cv2
import easyocr
import keras_ocr
import numpy as np
import pytesseract

from .utils import TextBox
from ..base import UiElement


class BaseOCR:
    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def recognize(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[str, List[str]]:
        raise NotImplementedError()

    @abstractmethod
    def _get_text_box(self, result: Any) -> TextBox:
        raise NotImplementedError()

    def process(self, elements: List[UiElement]):
        images = [e.get_cropped_image() for e in elements]
        texts = self.recognize(images)
        for e, text in zip(elements, texts):
            e.info["text"] = text

    def _merge_results(self, results: Iterable[Any]) -> List[TextBox]:
        def sort_key(text_box: TextBox):
            x0, y0 = text_box.bbox()[0]
            return y0, x0

        text_boxes = sorted(
            (self._get_text_box(result) for result in results),
            key=sort_key,
        )

        lines: List[Tuple[TextBox, List[TextBox]]] = []
        for text_box in text_boxes:
            if len(lines) == 0 or not lines[-1][0].is_on_same_line(text_box):
                lines.append((text_box, [text_box]))
            else:
                lines[-1] = (
                    lines[-1][0].merge(text_box, (lambda *_: None)),
                    lines[-1][1] + [text_box],
                )
        return [
            TextBox.merge_boxes(line, (lambda s0, s1: s0 + " " + s1))
            for _, line in lines
        ]


class EasyOCR(BaseOCR):
    model: easyocr.Reader

    def __init__(self):
        super().__init__(easyocr.Reader(["en"]))

    def _recognize(self, image: np.ndarray) -> str:
        results = self._merge_results(self.model.readtext(image))
        return "\n".join(result.text for result in results)

    def recognize(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[str, List[str]]:
        if isinstance(images, list):
            return [self._recognize(image) for image in images]
        return self._recognize(images)

    def _get_text_box(self, result: Any) -> TextBox:
        box = tuple(tuple(v) for v in result[0])
        text = result[1]
        return TextBox(box, text)


class KerasOCR(BaseOCR):
    model: keras_ocr.pipeline.Pipeline

    def __init__(self):
        super().__init__(keras_ocr.pipeline.Pipeline())

    def recognize(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[str, List[str]]:
        if isinstance(images, list):
            # results = self.model.recognize(images)
            results = []
            for image in images:
                results.extend(self.model.recognize([image]))
        else:
            results = self.model.recognize([images])
        results = [
            "\n".join(r.text for r in self._merge_results(result)) for result in results
        ]
        if len(results) == 1:
            return results[0]
        return results

    @staticmethod
    def _get_text_box(result: Any) -> TextBox:
        return TextBox(tuple(tuple(row) for row in result[1]), result[0])


class TesseractOCR(BaseOCR):
    model: pytesseract

    def __init__(self):
        super().__init__(pytesseract)

    def recognize(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[str, List[str]]:
        if isinstance(images, list):
            return [
                pytesseract.image_to_string(
                    cv2.threshold(
                        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                        127,
                        255,
                        cv2.THRESH_OTSU,
                    )[1]
                ).strip()
                for image in images
            ]
        return pytesseract.image_to_string(
            cv2.threshold(
                cv2.cvtColor(images, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_OTSU
            )[1]
        ).strip()
