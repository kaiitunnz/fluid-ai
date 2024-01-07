from typing import Any, List, Optional, Sequence

import keras_ocr  # type: ignore

from ..base import Array
from .models import BaseOCR
from .utils import TextBox


class KerasOCR(BaseOCR):
    """
    A text recognition module based on [keras-ocr](https://github.com/faustomorales/keras-ocr).

    Attributes
    ----------
    model : keras_ocr.pipeline.Pipeline
        Instance of keras-ocr's Pipeline.
    """

    model: keras_ocr.pipeline.Pipeline

    def __init__(self):
        super().__init__(keras_ocr.pipeline.Pipeline())

    def recognize(
        self,
        images: Sequence[Array],
    ) -> List[Optional[str]]:
        tmp = []
        for image in images:
            tmp.extend(self.model.recognize([image]))
        results: List[Optional[str]] = [
            "\n".join((r.text or "") for r in self._merge_results(result))
            for result in tmp
        ]
        return results

    @staticmethod
    def _get_text_box(result: Any) -> TextBox:
        return TextBox(tuple(tuple(row) for row in result[1]), result[0])  # type: ignore
