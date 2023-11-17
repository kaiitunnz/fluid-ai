import cv2
import pickle
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import gist  # type: ignore
import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore

from ..base import UiDetectionModule, UiElement
from .utils import compute_iou, cosine_similarity

# A hack, since linear_sum_assignment does not work with infinite values.
DISALLOWED = -sys.float_info.max


class BaseUiMatching(UiDetectionModule):
    @abstractmethod
    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        raise NotImplementedError()

    @abstractmethod
    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        raise NotImplementedError()

    def __call__(
        self, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        return self.match(base, other)


class IouUiMatching(BaseUiMatching):
    threshold: float

    def __init__(self, iou_threshold: float = 0.6):
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


class FeatureUiMatching(BaseUiMatching):
    iou_threshold: float
    similarity_threshold: float

    similarity_func: Callable[[np.ndarray, np.ndarray], float]

    def __init__(
        self,
        iou_threshold: float,
        similarity_threshold: float,
        similarity_func: Callable[[np.ndarray, np.ndarray], float],
    ):
        self.iou_threshold = iou_threshold
        self.similarity_threshold = similarity_threshold
        self.similarity_func = similarity_func

    @abstractmethod
    def _extract_feature(self, elem: UiElement) -> np.ndarray:
        raise NotImplementedError()

    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        if len(base) == 0:
            return other

        base_gist: List[Optional[np.ndarray]] = [None] * len(base)
        other_gist: List[Optional[np.ndarray]] = [None] * len(other)

        matched: List[Tuple[int, List[float]]] = []
        unmatched: List[UiElement] = []

        for o_idx, o_elem in enumerate(other):
            similarity_list = [DISALLOWED] * len(base)
            match_found = False
            for i, b_elem in enumerate(base):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.iou_threshold:
                    if base_gist[i] is None:
                        base_gist[i] = self._extract_feature(b_elem)
                    b_gist = base_gist[i]
                    if other_gist[o_idx] is None:
                        other_gist[o_idx] = self._extract_feature(o_elem)
                    o_gist = other_gist[o_idx]
                    assert not (b_gist is None or o_gist is None)
                    similarity = self.similarity_func(b_gist, o_gist)
                    if similarity > self.similarity_threshold:
                        match_found = True
                        similarity_list[i] = similarity
            if match_found:
                matched.append((o_idx, similarity_list))
            else:
                unmatched.append(o_elem)

        more_unmatched, _ = self._find_best_matches(matched)

        return base + unmatched + [other[i] for i in more_unmatched]

    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        """
        For debugging.
        """
        if len(base) == 0:
            return other

        base_gist: List[Optional[np.ndarray]] = [None] * len(base)
        other_gist: List[Optional[np.ndarray]] = [None] * len(other)

        matched: List[Tuple[int, List[float]]] = []
        unmatched: List[UiElement] = []

        for o_idx, o_elem in enumerate(other):
            similarity_list = [DISALLOWED] * len(base)
            match_found = False
            for i, b_elem in enumerate(base):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.iou_threshold:
                    if base_gist[i] is None:
                        base_gist[i] = self._extract_feature(b_elem)
                    b_gist = base_gist[i]
                    if other_gist[o_idx] is None:
                        other_gist[o_idx] = self._extract_feature(o_elem)
                    o_gist = other_gist[o_idx]
                    assert not (b_gist is None or o_gist is None)
                    similarity = self.similarity_func(b_gist, o_gist)
                    if similarity > self.similarity_threshold:
                        match_found = True
                        similarity_list[i] = similarity
            if match_found:
                matched.append((o_idx, similarity_list))
            else:
                unmatched.append(o_elem)

        more_unmatched, (base_matched, other_matched) = self._find_best_matches(matched)

        with open(f"res/base_matched{job_no}.pkl", "wb") as f:
            pickle.dump(base_matched, f)
        with open(f"res/other_matched{job_no}.pkl", "wb") as f:
            pickle.dump(other_matched, f)

        return base + unmatched + [other[i] for i in more_unmatched]

    def _find_best_matches(
        self, matched: List[Tuple[int, List[float]]]
    ) -> Tuple[List[int], Tuple[List[int], List[int]]]:
        """
        Return:
            unmatched_other_idxs, (matched_base_idxs, matched_other_idxs)
        """
        if len(matched) == 0:
            return [], ([], [])

        other_idxs, profit_matrix = zip(*matched)
        row_idx, col_idx = linear_sum_assignment(np.array(profit_matrix), maximize=True)

        unmatched_other_idxs = [
            other_idxs[i] for i in range(len(other_idxs)) if i not in row_idx
        ]
        matched_base_idxs, matched_other_idxs = [], []
        for i, j in zip(row_idx, col_idx):
            if profit_matrix[i][j] == DISALLOWED:
                unmatched_other_idxs.append(other_idxs[i])
            else:
                matched_other_idxs.append(other_idxs[i])
                matched_base_idxs.append(j)

        return unmatched_other_idxs, (matched_base_idxs, matched_other_idxs)


class HogUiMatching(FeatureUiMatching):
    win_size: Tuple[int, int]
    hog: cv2.HOGDescriptor

    def __init__(
        self,
        iou_threshold: float = 0.6,
        similarity_threshold: float = 0.2,
        win_size: Tuple[int, int] = (64, 64),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        cell_size: Tuple[int, int] = (8, 8),
        nbins: int = 9,
    ):
        super().__init__(iou_threshold, similarity_threshold, cosine_similarity)
        self.win_size = win_size
        self.hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=block_size,
            _blockStride=block_stride,
            _cellSize=cell_size,
            _nbins=nbins,
        )

    def _extract_feature(self, elem: UiElement) -> np.ndarray:
        converted = cv2.cvtColor(
            cv2.resize(elem.get_cropped_image(), self.win_size),  # type: ignore
            cv2.COLOR_RGB2GRAY,
        )  # type: ignore
        return np.array(self.hog.compute(converted))


class GistUiMatching(FeatureUiMatching):
    iou_threshold: float
    similarity_threshold: float
    size: Tuple[int, int]

    def __init__(
        self,
        iou_threshold: float = 0.3,
        similarity_threshold: float = 0.8,
        size: Tuple[int, int] = (64, 64),
    ):
        super().__init__(iou_threshold, similarity_threshold, cosine_similarity)
        self.size = size

    def _extract_feature(self, elem: UiElement) -> np.ndarray:
        resized = cv2.resize(elem.get_cropped_image(), self.size)  # type: ignore
        return gist.extract(resized)  # type: ignore
