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
    """
    A base class for UI element matching.

    A class that implements a UI matching model to be used in the UI detection pipeline
    must inherit this class.
    """

    @abstractmethod
    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        """Matches the UI elements in `other` to the UI elements in `base`

        A UI matching model must implement this method.

        Parameters
        ----------
        base : List[UiElement]
            The base list of UI elements. Each UI element in this list will be matched
            with each of those in `other`. The elements whose match is found will be
            included in the resulting list of UI elements.
        other : List[UiElement]
            The list of UI elements with which the base elements will be matched.
            If a match is found, the UI elements of this list will be discarded in
            favor of the UI elements in `base`.

        Returns
        -------
        List[UiElement]
            A resulting list of UI elements. It contains unique UI elements whose
            duplicate elements are discarded.
        """
        raise NotImplementedError()

    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        """Matches the UI elements in `other` to the UI elements in `base` given `job_no`

        It is used for debugging or benchmarking the UI detection pipeline. By default,
        it forwards `base` and `other` to `self.match`.

        Parameters
        ----------
        job_no : int
            ID of the current job.
        base : List[UiElement]
            The base list of UI elements. Each UI element in this list will be matched
            with each of those in `other`. The elements whose match is found will be
            included in the resulting list of UI elements.
        other : List[UiElement]
            The list of UI elements with which the base elements will be matched.
            If a match is found, the UI elements of this list will be discarded in
            favor of the UI elements in `base`.

        Returns
        -------
        List[UiElement]
            A resulting list of UI elements. It contains unique UI elements whose
            duplicate elements are discarded.
        """
        return self.match(base, other)

    def find_best_matches(
        self, matched: List[Tuple[int, List[float]]]
    ) -> Tuple[List[int], Tuple[List[int], List[int]]]:
        """Finds best profit matching based on the LSAP.

        Parameters
        ----------
        matched : List[Tuple[int, List[float]]]
            A list of pairs of ID and a list of profit values.

        Returns
        -------
        Tuple[List[int], Tuple[List[int], List[int]]]
            `unmatched_other_idxs, (matched_base_idxs, matched_other_idxs)`. `unmatched_other_idxs`
            is the list of indices of elements in `other` that have no match, based
            on the profit, i.e., similarity score. `matched_base_idxs` and `matched_other_idxs`
            are the lists of indices of elements in `base` and `other`, respectively,
            that match.
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

    def __call__(
        self, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        """Matches the UI elements in `other` to the UI elements in `base`

        A UI matching model must implement this method.

        Parameters
        ----------
        base : List[UiElement]
            The base list of UI elements. Each UI element in this list will be matched
            with each of those in `other`. The elements whose match is found will be
            included in the resulting list of UI elements.
        other : List[UiElement]
            The list of UI elements with which the base elements will be matched.
            If a match is found, the UI elements of this list will be discarded in
            favor of the UI elements in `base`.

        Returns
        -------
        List[UiElement]
            A resulting list of UI elements. It contains unique UI elements whose
            duplicate elements are discarded.
        """
        return self.match(base, other)


class DummyUiMatching(BaseUiMatching):
    """
    A dummy UI matching model.

    It returns a concatenated list of `base` and `other`.
    """

    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        return base + other

    def match_i(
        self, _: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        return base + other


class IouUiMatching(BaseUiMatching):
    """
    A UI matching model based on Intersection over Union (IoU).

    Attributes
    ----------
    threshold : float
        The threshold value to identify matching UI elements. Matching elements'
        IOU values must exceed this threshold.
    """

    threshold: float

    def __init__(self, iou_threshold: float = 0.6):
        """
        Parameters
        ----------
        iou_threshold : float
            The threshold value to identify matching UI elements. Matching elements'
            IoU values must exceed this threshold. Best matches are compute
        """
        self.threshold = iou_threshold

    def match(self, base: List[UiElement], other: List[UiElement]) -> List[UiElement]:
        if len(base) == 0:
            return other

        matched: List[Tuple[int, List[float]]] = []
        unmatched: List[UiElement] = []

        for o_idx, o_elem in enumerate(other):
            iou_list = [DISALLOWED] * len(base)
            match_found = False
            for i, b_elem in enumerate(base):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.threshold:
                    match_found = True
                    iou_list[i] = iou
            if match_found:
                matched.append((o_idx, iou_list))
            else:
                unmatched.append(o_elem)

        more_unmatched, _ = self.find_best_matches(matched)

        return base + unmatched + [other[i] for i in more_unmatched]

    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
        if len(base) == 0:
            return other

        matched: List[Tuple[int, List[float]]] = []
        unmatched: List[UiElement] = []

        for o_idx, o_elem in enumerate(other):
            iou_list = [DISALLOWED] * len(base)
            match_found = False
            for i, b_elem in enumerate(base):
                iou = compute_iou(b_elem.bbox, o_elem.bbox)
                if iou > self.threshold:
                    match_found = True
                    iou_list[i] = iou
            if match_found:
                matched.append((o_idx, iou_list))
            else:
                unmatched.append(o_elem)

        more_unmatched, (base_matched, other_matched) = self.find_best_matches(matched)

        with open(f"res/base_matched{job_no}.pkl", "wb") as f:
            pickle.dump(base_matched, f)
        with open(f"res/other_matched{job_no}.pkl", "wb") as f:
            pickle.dump(other_matched, f)

        return base + unmatched + [other[i] for i in more_unmatched]


class FeatureUiMatching(BaseUiMatching):
    """
    A base class for a UI matching model based on feature extraction.

    A class that implements a feature-based UI matching model must inherit this
    class.

    Attributes
    ----------
    iou_threshold : float
        The IoU threshold value to identify matching UI elements. Matching elements'
        IoU values must exceed this threshold.
    similarity_threshold : float
        The threshold value for similarity score to identify matching UI elements.
        Matching elements' similarity scores must exceed this threshold.
    similarity_func : Callable[[np.ndarray, np.ndarray], float]
        A function to calculate the similarity score from two NumPy arrays.
    """

    iou_threshold: float
    similarity_threshold: float

    similarity_func: Callable[[np.ndarray, np.ndarray], float]

    def __init__(
        self,
        iou_threshold: float,
        similarity_threshold: float,
        similarity_func: Callable[[np.ndarray, np.ndarray], float],
    ):
        """
        Parameters
        ----------
        iou_threshold : float
            The IoU threshold value to identify matching UI elements. Matching elements'
            IoU values must exceed this threshold.
        similarity_threshold : float
            The threshold value for similarity score to identify matching UI elements.
            Matching elements' similarity scores must exceed this threshold.
        similarity_func : Callable[[np.ndarray, np.ndarray], float]
            A function to calculate the similarity score from two NumPy arrays.
        """
        self.iou_threshold = iou_threshold
        self.similarity_threshold = similarity_threshold
        self.similarity_func = similarity_func

    @abstractmethod
    def _extract_feature(self, elem: UiElement) -> np.ndarray:
        """Extracts a feature vector from a `UiElement`

        A feature-based UI matching model must implement this method.

        Parameters
        ----------
        elem : UiElement
            A `UiElement` to extract a feature vector.
        """
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

        more_unmatched, _ = self.find_best_matches(matched)

        return base + unmatched + [other[i] for i in more_unmatched]

    def match_i(
        self, job_no: int, base: List[UiElement], other: List[UiElement]
    ) -> List[UiElement]:
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

        more_unmatched, (base_matched, other_matched) = self.find_best_matches(matched)

        with open(f"res/base_matched{job_no}.pkl", "wb") as f:
            pickle.dump(base_matched, f)
        with open(f"res/other_matched{job_no}.pkl", "wb") as f:
            pickle.dump(other_matched, f)

        return base + unmatched + [other[i] for i in more_unmatched]


class HogUiMatching(FeatureUiMatching):
    """
    A class that implement a UI matching model based on HOG feature descriptors.

    Attributes
    ----------
    win_size : Tuple[int, int]
        Window size
    hog : HOGDescriptor
        The object that extracts a HOG descriptor from a NumPy array.
    """

    win_size: Tuple[int, int]
    hog: cv2.HOGDescriptor

    def __init__(
        self,
        iou_threshold: float = 0.3,
        similarity_threshold: float = 0.4,
        win_size: Tuple[int, int] = (64, 64),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        cell_size: Tuple[int, int] = (8, 8),
        nbins: int = 9,
    ):
        """
        Parameters
        ----------
        iou_threshold : float
            The IoU threshold value to identify matching UI elements. Matching elements'
            IoU values must exceed this threshold.
        similarity_threshold : float
            The threshold value for similarity score to identify matching UI elements.
            Matching elements' similarity scores must exceed this threshold.
        win_size : Tuple[int, int]
            Window size
        block_size : Tuple[int, int]
            See [this](https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html#ac0544de0ddd3d644531d2164695364d9)
            for more details.
        block_stride : Tuple[int, int]
            See [this](https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html#ac0544de0ddd3d644531d2164695364d9)
            for more details.
        cell_size : Tuple[int, int]
            See [this](https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html#ac0544de0ddd3d644531d2164695364d9)
            for more details.
        nbins : int
            Number of bins in the gradient histogram. See [this](https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html#ac0544de0ddd3d644531d2164695364d9)
            for more details.
        """
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
    """
    A base class for a UI matching model based on feature extraction.

    A class that implements a feature-based UI matching model must inherit this
    class.

    Attributes
    ----------
    iou_threshold : float
        The IoU threshold value to identify matching UI elements. Matching elements'
        IoU values must exceed this threshold.
    similarity_threshold : float
        The threshold value for similarity score to identify matching UI elements.
        Matching elements' similarity scores must exceed this threshold.
    similarity_func : Callable[[np.ndarray, np.ndarray], float]
        A function to calculate the similarity score from two NumPy arrays.
    size : Tuple[int, int]
        Size (w, h) to which the input UI element image is resized.
    """

    size: Tuple[int, int]

    def __init__(
        self,
        iou_threshold: float = 0.3,
        similarity_threshold: float = 0.8,
        size: Tuple[int, int] = (64, 64),
    ):
        """
        Parameters
        ----------
        iou_threshold : float
            The IoU threshold value to identify matching UI elements. Matching elements'
            IoU values must exceed this threshold.
        similarity_threshold : float
            The threshold value for similarity score to identify matching UI elements.
            Matching elements' similarity scores must exceed this threshold.
        size : Tuple[int, int]
            Size (w, h) to which the input UI element image is resized.
        """
        super().__init__(iou_threshold, similarity_threshold, cosine_similarity)
        self.size = size

    def _extract_feature(self, elem: UiElement) -> np.ndarray:
        resized = cv2.resize(elem.get_cropped_image(), self.size)  # type: ignore
        return gist.extract(resized)  # type: ignore
