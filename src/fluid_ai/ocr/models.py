from abc import abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import easyocr  # type: ignore
import pytesseract  # type: ignore

from .utils import TextBox
from ..base import Array, UiDetectionModule, UiElement


class BaseOCR(UiDetectionModule):
    """
    A base class for text recognition modules.

    A class that implements a text recognition module to be used in the UI detection
    pipeline must inherit this class.

    Attributes
    ----------
    model : Any
        Text recognition model.
    """

    model: Any

    def __init__(self, model: Any):
        """
        Parameters
        ----------
        model : Any
            Text recognition model.
        """
        self.model = model

    @abstractmethod
    def recognize(
        self,
        images: Sequence[Array],
    ) -> List[Optional[str]]:
        """Recognizes texts from images

        Parameters
        ----------
        images : Sequence[Array]
            List of input images.

        Returns
        -------
        List[Optional[str]]
            List of recognized texts. None if not applicable.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_text_box(self, result: Any) -> TextBox:
        """Creates a text box for the recognition result

        Parameters
        ----------
        result : Any
            Recognition result.

        Returns
        -------
        TextBox
            Created text box.
        """
        raise NotImplementedError()

    def __call__(
        self,
        elements: List[UiElement],
        loader: Optional[Callable[..., Array]] = None,
    ):
        """Recognizes texts in the input UI elements

        This function modifies the input UI elements. The recognized text of each
        UI element, `element`, is stored with a "text" key in `element.info`.

        Parameters
        ----------
        elements : List[UiElement]
            List of input UI elements.
        loader : Optional[Callable[..., Array]]
            Image loader, used to load the screenshot images of the UI elements.
        """
        self.process(elements, loader)

    def process(
        self,
        elements: List[UiElement],
        loader: Optional[Callable[..., Array]] = None,
    ):
        """Recognizes texts in the input UI elements

        This function modifies the input UI elements. The recognized text of each
        UI element, `element`, is stored with a "text" key in `element.info`.

        Parameters
        ----------
        elements : List[UiElement]
            List of input UI elements.
        loader : Optional[Callable[..., Array]]
            Image loader, used to load the screenshot images of the UI elements.
        """
        images = [e.get_cropped_image(loader) for e in elements]
        texts = self.recognize(images)
        for e, text in zip(elements, texts):
            if text is None:
                continue
            e.info["text"] = text

    def _merge_results(self, results: Iterable[Any]) -> List[TextBox]:
        """Merges the recognition results that appear on the same lines

        Parameters
        ----------
        results : Iterable[Any]
            List of recognition results.

        Returns
        -------
        List[TextBox]
            List of merged text boxes.
        """

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
                    lines[-1][0].merge(text_box, (lambda *_: "")),
                    lines[-1][1] + [text_box],
                )
        return [
            TextBox.merge_boxes(line, (lambda s0, s1: s0 + " " + s1))
            for _, line in lines
        ]


class DummyOCR(BaseOCR):
    """
    A dummy text recognition module.

    It returns invalid texts for all input images.

    Attributes
    ----------
    model : Any
        Text recognition model (set to None).
    """

    model: None

    def __init__(self):
        super().__init__(None)

    def recognize(
        self,
        images: Sequence[Array],
    ) -> List[Optional[str]]:
        return [None] * len(images)

    def _get_text_box(self, _: Any) -> TextBox:
        box = ((0, 0), (0, 0), (0, 0), (0, 0))
        return TextBox(box)


class EasyOCR(BaseOCR):
    """
    A text recognition module based on [EasyOCR](https://github.com/JaidedAI/EasyOCR).

    Attributes
    ----------
    model : easyocr.Reader
        Instance of EasyOCR's Reader.
    batch_size : int
        Inference batch size. Recommend setting to 1 for accurate results.
    image_size : Tuple[int, int]
        Size `(width, height)` to which the images will be resized when performing
        batch inference.
    """

    model: easyocr.Reader
    batch_size: int
    image_size: Tuple[int, int]

    def __init__(self, batch_size: int = 1, image_size: Tuple[int, int] = (224, 224)):
        """
        Parameters
        ----------
        batch_size : int
            Inference batch size. Recommend setting to 1 for accurate results.
        image_size : Tuple[int, int]
            Size `(width, height)` to which the images will be resized when performing
            batch inference.
        """
        if batch_size < 1:
            raise ValueError("Invalid batch size.")
        self.batch_size = batch_size
        self.image_size = image_size
        super().__init__(easyocr.Reader(["en"]))

    def _recognize(self, image: Array) -> str:
        """Recognizes text in an image

        Parameters
        ----------
        image : Array
            Input image.

        Returns
        -------
        str
            Recognized text.
        """
        results = self._merge_results(self.model.readtext(image))
        return "\n".join((result.text or "") for result in results)

    def _recognize_batched(self, images: Sequence[Array]) -> List[Optional[str]]:
        """Recognizes texts in images with batch inference

        Parameters
        ----------
        image : Sequence[Array]
            List of input images.

        Returns
        -------
        str
            List of recognized texts.
        """
        unmerged = self.model.readtext_batched(
            images,
            n_width=self.image_size[0],
            n_height=self.image_size[1],
            batch_size=self.batch_size,
        )
        results = [self._merge_results(each) for each in unmerged]

        return ["\n".join((box.text or "") for box in result) for result in results]

    def recognize(
        self,
        images: Sequence[Array],
    ) -> List[Optional[str]]:
        if self.batch_size <= 1:
            return [self._recognize(image) for image in images]
        return self._recognize_batched(images)

    def _get_text_box(self, result: Any) -> TextBox:
        box = tuple(tuple(v) for v in result[0])
        text = result[1]
        return TextBox(box, text)  # type: ignore


class KerasOCR(BaseOCR):
    """
    A text recognition module based on [keras-ocr](https://github.com/faustomorales/keras-ocr).

    Attributes
    ----------
    model : keras_ocr.pipeline.Pipeline
        Instance of keras-ocr's Pipeline.
    """

    pass


class TesseractOCR(BaseOCR):
    """
    A text recognition module based on [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

    Attributes
    ----------
    model : Module
        Python-tesseract module.
    """

    model: pytesseract  # type: ignore

    def __init__(self):
        super().__init__(pytesseract)

    def recognize(
        self,
        images: Sequence[Array],
    ) -> List[Optional[str]]:
        return [
            pytesseract.image_to_string(
                cv2.threshold(
                    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),  # type: ignore
                    127,
                    255,
                    cv2.THRESH_OTSU,
                )[
                    1
                ]  # type: ignore
            ).strip()
            for image in images
        ]
