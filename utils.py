import random
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.yolo.engine.results import Annotator

from .base import UiElement


def _compute_luminance(r: int, g: int, b: int) -> float:
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


class _PlotColor:
    bg: Tuple[int, int, int]
    text: Tuple[int, int, int]

    def __init__(self, bg: Tuple[int, int, int]):
        self.bg = bg
        self.text = (0, 0, 0) if _compute_luminance(*bg) > 0.5 else (255, 255, 255)


def _get_color_map(elem: List[UiElement]) -> Dict[str, _PlotColor]:
    elem_classes = set(e.name for e in elem)
    color_map = {}
    for elem_class in elem_classes:
        bg_color = tuple(random.choices(range(256), k=3))
        while bg_color in color_map:
            bg_color = tuple(random.choices(range(256), k=3))
        color_map[elem_class] = _PlotColor(bg_color)
    return color_map


def plot_ui_elements(img: np.ndarray, elem: List[UiElement], scale: float = 1.0):
    plt.xticks([], [])
    plt.yticks([], [])
    color_map = _get_color_map(elem)
    h, w, *_ = img.shape
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    annotator = Annotator(img)
    for i, e in enumerate(elem, 1):
        color = color_map[e.name]
        (x0, y0), (x1, y1) = e.bbox
        bbox = tuple(int(e * scale) for e in (x0, y0, x1, y1))
        annotator.box_label(bbox, f"({i})", color.bg, color.text)
    plt.imshow(annotator.result())
    plt.show()
