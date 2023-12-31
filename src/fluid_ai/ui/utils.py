import numpy as np

from ..base import BaseBBox


def compute_iou(bbox1: BaseBBox, bbox2: BaseBBox) -> float:
    """Computes Intersection over Union (IoU) value of the given bounding boxes

    Parameters
    ----------
    bbox1 : BaseBBox
        First bounding box.
    bbox2 : BaseBBox
        Second bounding box.

    Returns
    -------
    float
        IoU value
    """
    (x0_1, y0_1), (x1_1, y1_1) = bbox1
    (x0_2, y0_2), (x1_2, y1_2) = bbox2

    x0_intersect = max(x0_1, x0_2)
    y0_intersect = max(y0_1, y0_2)
    x1_intersect = min(x1_1, x1_2)
    y1_intersect = min(y1_1, y1_2)

    intersect_area = max(0, x1_intersect - x0_intersect) * max(
        0, y1_intersect - y0_intersect
    )

    area_box1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area_box2 = (x1_2 - x0_2) * (y1_2 - y0_2)

    union_area = area_box1 + area_box2 - intersect_area

    if union_area == 0:
        return 0.0
    return intersect_area / union_area


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Computes cosine similarity between two vectors

    Parameters
    ----------
    u : ndarray
        First vector
    v : ndarray
        Second vector

    Returns
    -------
    float
        Cosine similarity value
    """
    return np.dot(u, v) / (np.sqrt(np.sum(u**2) * np.sum(v**2)))  # type: ignore
