from typing import Callable, Iterable, NamedTuple, Optional

from typing_extensions import Self

from ..base import BBox, Box


class TextBox(NamedTuple):
    """
    Text box.

    This class provides functionalities for manipulating text boxes.

    Attributes
    ----------
    box : Box
        Bounding box of the text. Unlike `BBox`, it may not be rectangular.
    text : Optional[str]
        Text.
    """

    box: Box
    text: Optional[str] = None

    def bbox(self) -> BBox:
        """Gets a rectangular bounding box (`BBox`) of the text

        Returns
        -------
        BBox
            Rectangular bounding box of the text.
        """
        xs, ys = zip(*self.box)
        return BBox((min(xs), min(ys)), (max(xs), max(ys)))

    def is_on_same_line(self, other: Self) -> bool:
        """Returns whether the text represented by this text box is on the same line
        as `other`'s

        Parameters
        ----------
        other : TextBox
            Text box to be compared.

        Returns
        -------
        bool
            Whether the text represented by this text box is on the same line as `other`'s.
        """
        (_, y10), (_, y11) = self.bbox()
        (_, y20), (_, y21) = other.bbox()
        return (
            (y10 <= y20 <= (y10 + y11) / 2)
            or ((y10 + y11) / 2 <= y21 <= y11)
            or (y21 >= y11 and y20 <= y10)
        )

    def merge(self, other: Self, merge_func: Callable[[str, str], str]) -> "TextBox":
        """Merges this text box with `other`

        The texts in this text box and `other` are merged with `merge_func`.

        Parameters
        ----------
        other : TextBox
            Text box to be merged with.
        merge_func : Callable[[str, str], str]
            Function to merge the texts in the text boxes.

        Returns
        -------
        TextBox
            Merged text box.
        """
        (x00, y00), (x01, y01) = self.bbox()
        (x10, y10), (x11, y11) = other.bbox()
        x0, y0, x1, y1 = min(x00, x10), min(y00, y10), max(x01, x11), max(y01, y11)
        box = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
        if self.text is None:
            text = other.text
        elif other.text is None:
            text = self.text
        else:
            if x00 < x10:
                text = merge_func(self.text, other.text)
            else:
                text = merge_func(other.text, self.text)
        return TextBox(box, text)

    @classmethod
    def merge_boxes(
        cls, text_boxes: Iterable[Self], merge_func: Callable[[str, str], str]
    ) -> Self:
        """Merge multiple text boxes

        The texts in the input text boxes are merged with `merge_func` in a "fold-left"
        manner.

        Parameters
        ----------
        text_boxes : Iterable[TextBox]
            Text boxes to be merged. They should be appropriately ordered.
        merge_func : Callable[[str, str], str]
            Function to merge the texts in the text boxes.

        Returns
        -------
        TextBox
            Merged text box.
        """
        xs, ys = zip(*(v for text_box in text_boxes for v in text_box.bbox()))
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        box = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
        text_boxes = sorted(text_boxes, key=(lambda b: b.bbox()[0][0]))
        text = None
        for text_box in text_boxes:
            if text_box.text is None:
                continue
            if text is None:
                text = text_box.text
            else:
                text = merge_func(text, text_box.text)
        return cls(box, text)
