import random
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from ultralytics.engine.results import Annotator  # type: ignore

from .base import UiElement


def _compute_luminance(r: int, g: int, b: int) -> float:
    """Computes the luminance of the given color

    Parameters
    ----------
    r : int
        Red channel in the interval [0, 255].
    g : int
        Green channel in the interval [0, 255].
    b : int
        Blue channel in the interval [0, 255].

    Returns
    -------
    float
        Luminance value.
    """
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


class _PlotColor:
    """
    A utility class representing the appropriate colors for the background and
    foreground.

    Attributes
    ----------
    bg : Tuple[int, int, int]
        Background color.
    text : Tuple[int, int, int]
        Foreground color.
    """

    bg: Tuple[int, int, int]
    text: Tuple[int, int, int]

    def __init__(self, bg: Tuple[int, int, int]):
        """
        Parameters
        ----------
        bg : Tuple[int, int, int]
            Background color.
        """
        self.bg = bg
        self.text = (0, 0, 0) if _compute_luminance(*bg) > 0.5 else (255, 255, 255)


def _get_color_map(elems: List[UiElement]) -> Dict[str, _PlotColor]:
    """Gets a random mapping from UI names to plot colors

    Parameters
    ----------
    elems : List[UiElement]
        List of sample UI elements to create a mapping.

    Returns
    -------
    Dict[str, _PlotColor]
        Mapping from UI names to plot colors.
    """
    elem_classes = set(e.name for e in elems)
    color_map = {}
    for elem_class in elem_classes:
        bg_color = tuple(random.choices(range(256), k=3))
        while bg_color in color_map:
            bg_color = tuple(random.choices(range(256), k=3))
        color_map[elem_class] = _PlotColor(bg_color)  # type: ignore
    return color_map


def plot_ui_elements(
    img: np.ndarray,
    elems: List[UiElement],
    scale: float = 1.0,
    fname: Optional[str] = None,
):
    """Highlights UI elements on the given image according to their bounding boxes
    and displays the result.

    The UI elements are highlighted with rectangles corresponding to their bounding
    boxes with different colors representing different UI classes/names.

    Parameters
    ----------
    img : ndarray
        Screenshot containing the given UI elements.
    elems : List[UiElement]
        List of UI elements to be highlighted on the screenshot.
    scale : float
        Factor to scale the screenshot. The resulting image will be of size
        scale * size of the screenshot.
    save : bool
        Whether to save the resulting image.
    """
    plt.xticks([], [])
    plt.yticks([], [])
    color_map = _get_color_map(elems)
    h, w, *_ = img.shape
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    annotator = Annotator(img)
    for i, e in enumerate(elems, 1):
        color = color_map[e.name]
        (x0, y0), (x1, y1) = e.bbox
        bbox = tuple(int(e * scale) for e in (x0, y0, x1, y1))
        annotator.box_label(bbox, f"({i})", color.bg, color.text)
    if fname is None:
        plt.imshow(annotator.result())
        plt.show()
    else:
        plt.imsave(fname, annotator.result())
