import cv2
import pickle
from abc import abstractmethod
from typing import List, Optional, Tuple

import gist  # type: ignore
import numpy as np

from ..base import UiDetectionModule, UiElement
from .utils import compute_iou, cosine_similarity


class BaseUiMatching(UiDetectionModule):
    @abstractmethod
    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        raise NotImplementedError()

    def __call__(
        self, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        return self.match(base, other)


class IouUiMatching(BaseUiMatching):
    threshold: float

    def __init__(self, iou_threshold: float = 0.8):
        self.threshold = iou_threshold

    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        if len(base) == 0:
            return other

        unmatched = []
        tmp = base

        for o_elem in other:
            matched: List[Tuple[float, int]] = []
            for i, b_elem in enumerate(tmp):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.threshold:
                    matched.append((iou, i))
            if len(matched) == 0:
                unmatched.append(o_elem)
            else:
                _, best_i = max(matched)
                tmp = tmp[:best_i] + tmp[best_i + 1 :]

        return base + unmatched

    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        """
        For debugging.
        """
        if len(base) == 0:
            return other

        base_matched = []
        other_matched = []

        unmatched = []
        tmp = list(enumerate(base))

        for o_idx, o_elem in enumerate(other):
            matched: List[Tuple[float, int, int]] = []
            for j, (i, b_elem) in enumerate(tmp):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.threshold:
                    matched.append((iou, i, j))
            if len(matched) == 0:
                unmatched.append(o_elem)
            else:
                _, best_i, best_j = max(matched)
                base_matched.append(best_i)
                other_matched.append(o_idx)
                tmp = tmp[:best_j] + tmp[best_j + 1 :]

        with open(f"res/base_matched{job_no}.pkl", "wb") as f:
            pickle.dump(base_matched, f)
        with open(f"res/other_matched{job_no}.pkl", "wb") as f:
            pickle.dump(other_matched, f)

        return base + unmatched


class HogUiMatching(BaseUiMatching):
    iou_threshold: float
    similarity_threshold: float
    win_size: Tuple[int, int]
    hog: cv2.HOGDescriptor

    def __init__(
        self,
        iou_threshold: float = 0.2,
        similarity_threshold: float = 0.6,
        win_size: Tuple[int, int] = (64, 64),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        cell_size: Tuple[int, int] = (8, 8),
        nbins: int = 9,
    ):
        self.iou_threshold = iou_threshold
        self.similarity_threshold = similarity_threshold
        self.win_size = win_size
        self.hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=block_size,
            _blockStride=block_stride,
            _cellSize=cell_size,
            _nbins=nbins,
        )

    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        if len(base) == 0:
            return other

        base_gist: List[Optional[np.ndarray]] = [None] * len(base)
        other_gist: List[Optional[np.ndarray]] = [None] * len(other)

        unmatched = []
        tmp = list(enumerate(base))

        for o_idx, o_elem in enumerate(other):
            matched: List[Tuple[float, int]] = []
            for j, (i, b_elem) in enumerate(tmp):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.iou_threshold:
                    if base_gist[i] is None:
                        base_gist[i] = self._get_descriptor(b_elem)
                    b_gist = base_gist[i]
                    if other_gist[o_idx] is None:
                        other_gist[o_idx] = self._get_descriptor(o_elem)
                    o_gist = other_gist[o_idx]
                    assert not (b_gist is None or o_gist is None)
                    similarity = cosine_similarity(b_gist, o_gist)
                    if similarity > self.similarity_threshold:
                        matched.append((similarity, j))
            if len(matched) == 0:
                unmatched.append(o_elem)
            else:
                _, best_j = max(matched)
                tmp = tmp[:best_j] + tmp[best_j + 1 :]

        return base + unmatched

    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        """
        For debugging.
        """
        if len(base) == 0:
            return other

        base_matched = []
        other_matched = []

        base_gist: List[Optional[np.ndarray]] = [None] * len(base)
        other_gist: List[Optional[np.ndarray]] = [None] * len(other)

        unmatched = []
        tmp = list(enumerate(base))

        for o_idx, o_elem in enumerate(other):
            matched: List[Tuple[float, int, int]] = []
            for j, (i, b_elem) in enumerate(tmp):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.iou_threshold:
                    if base_gist[i] is None:
                        base_gist[i] = self._get_descriptor(b_elem)
                    b_gist = base_gist[i]
                    if other_gist[o_idx] is None:
                        other_gist[o_idx] = self._get_descriptor(o_elem)
                    o_gist = other_gist[o_idx]
                    assert not (b_gist is None or o_gist is None)
                    similarity = cosine_similarity(b_gist, o_gist)
                    if similarity > self.similarity_threshold:
                        matched.append((similarity, i, j))
            if len(matched) == 0:
                unmatched.append(o_elem)
            else:
                _, best_i, best_j = max(matched)
                base_matched.append(best_i)
                other_matched.append(o_idx)
                tmp = tmp[:best_j] + tmp[best_j + 1 :]
        with open(f"res/base_matched{job_no}.pkl", "wb") as f:
            pickle.dump(base_matched, f)
        with open(f"res/other_matched{job_no}.pkl", "wb") as f:
            pickle.dump(other_matched, f)

        return base + unmatched

    def _get_descriptor(self, elem: UiElement) -> np.ndarray:
        converted = cv2.cvtColor(
            cv2.resize(elem.get_cropped_image(), self.win_size),  # type: ignore
            cv2.COLOR_RGB2GRAY,
        )  # type: ignore
        return np.array(self.hog.compute(converted))


class GistUiMatching(BaseUiMatching):
    iou_threshold: float
    similarity_threshold: float
    size: Tuple[int, int]

    def __init__(
        self,
        iou_threshold: float = 0.2,
        similarity_threshold: float = 0.8,
        size: Tuple[int, int] = (64, 64),
    ):
        self.iou_threshold = iou_threshold
        self.similarity_threshold = similarity_threshold
        self.size = size

    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        if len(base) == 0:
            return other

        base_gist: List[Optional[np.ndarray]] = [None] * len(base)
        other_gist: List[Optional[np.ndarray]] = [None] * len(other)

        unmatched = []
        tmp = list(enumerate(base))

        for o_idx, o_elem in enumerate(other):
            matched: List[Tuple[float, int]] = []
            for j, (i, b_elem) in enumerate(tmp):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.iou_threshold:
                    if base_gist[i] is None:
                        base_gist[i] = self._get_descriptor(b_elem)
                    b_gist = base_gist[i]
                    if other_gist[o_idx] is None:
                        other_gist[o_idx] = self._get_descriptor(o_elem)
                    o_gist = other_gist[o_idx]
                    assert not (b_gist is None or o_gist is None)
                    similarity = cosine_similarity(b_gist, o_gist)
                    if similarity > self.similarity_threshold:
                        matched.append((similarity, j))
            if len(matched) == 0:
                unmatched.append(o_elem)
            else:
                _, best_j = max(matched)
                tmp = tmp[:best_j] + tmp[best_j + 1 :]

        return base + unmatched

    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        """
        For debugging.
        """
        if len(base) == 0:
            return other

        base_matched = []
        other_matched = []

        base_gist: List[Optional[np.ndarray]] = [None] * len(base)
        other_gist: List[Optional[np.ndarray]] = [None] * len(other)

        unmatched = []
        tmp = list(enumerate(base))

        for o_idx, o_elem in enumerate(other):
            matched: List[Tuple[float, int, int]] = []
            for j, (i, b_elem) in enumerate(tmp):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.iou_threshold:
                    if base_gist[i] is None:
                        base_gist[i] = self._get_descriptor(b_elem)
                    b_gist = base_gist[i]
                    if other_gist[o_idx] is None:
                        other_gist[o_idx] = self._get_descriptor(o_elem)
                    o_gist = other_gist[o_idx]
                    assert not (b_gist is None or o_gist is None)
                    similarity = cosine_similarity(b_gist, o_gist)
                    if similarity > self.similarity_threshold:
                        matched.append((similarity, i, j))
            if len(matched) == 0:
                unmatched.append(o_elem)
            else:
                _, best_i, best_j = max(matched)
                base_matched.append(best_i)
                other_matched.append(o_idx)
                tmp = tmp[:best_j] + tmp[best_j + 1 :]

        with open(f"res/base_matched{job_no}.pkl", "wb") as f:
            pickle.dump(base_matched, f)
        with open(f"res/other_matched{job_no}.pkl", "wb") as f:
            pickle.dump(other_matched, f)

        return base + unmatched

    def _get_descriptor(self, elem: UiElement) -> np.ndarray:
        resized = cv2.resize(elem.get_cropped_image(), self.size)  # type: ignore
        return gist.extract(resized)  # type: ignore
