from abc import abstractmethod
from typing import List

from ..base import UiDetectionModule, UiElement
from .utils import compute_iou


class BaseUiRelation(UiDetectionModule):
    """
    A base class for UI relation.

    A class that implements a UI relation module to be used in the UI detection pipeline
    must inherit this class.

    Attributes
    ----------
    RELATION : str
        Relation name.
    """

    RELATION: str = "base"

    @abstractmethod
    def relate(self, elements: List[UiElement], relation: str):
        """Extracts the relation between UI elements

        This function modifies the input UI elements. The relation of each UI element,
        `element`, is stored with a "<relation>" key in `element.relation`, where
        "<relation>" is the relation name.

        Parameters
        ----------
        elements : List[UiElement]
            List of UI elements.
        relation : List[UiElement]
            Relation name.
        """
        raise NotImplementedError()

    def __call__(self, elements: List[UiElement]):
        """Extracts the relation between UI elements

        This function modifies the input UI elements. The relation of each UI element,
        `element`, is stored with a "<relation>" key in `element.relation`, where
        "<relation>" is the relation name.

        Parameters
        ----------
        elements : List[UiElement]
            List of UI elements.
        relation : List[UiElement]
            Relation name.
        """
        self.relate(elements, self.RELATION)


class DummyUiRelation(BaseUiRelation):
    """
    A dummy UI relation module.

    It returns the input elements.

    Attributes
    ----------
    RELATION : str
        Relation name.
    """

    RELATION: str = "dummy"

    def relate(self, elements: List[UiElement], _: str):
        """Extracts the relation between UI elements

        It does nothing.

        Parameters
        ----------
        elements : List[UiElement]
            List of UI elements.

        Returns
        -------
        List[UiElement]
            List of UI elements.
        """
        pass


class UiOverlap(BaseUiRelation):
    """
    Overlapping UI relation model.

    Attributes
    ----------
    iou_threshold : float
        The IoU threshold value to identify overlapping UI elements. Overlapping elements'
        IoU values must exceed this threshold.
    RELATION : str
        Relation name.
    """

    iou_threshold: float
    RELATION: str = "overlap"

    def __init__(self, iou_threshold: float = 0):
        self.iou_threshold = iou_threshold

    def relate(self, elements: List[UiElement], relation: str):
        total = len(elements)

        for i in range(total):
            i_elem = elements[i]
            if i_elem.relation.get(relation) is None:
                i_elem.relation[relation] = []
            for j in range(i + 1, total):
                j_elem = elements[j]
                iou_value = compute_iou(i_elem.bbox, j_elem.bbox)
                if iou_value > self.iou_threshold:
                    i_elem.relation[relation].append(j)
                    j_elem_overlap = j_elem.relation.get(relation)
                    if isinstance(j_elem_overlap, list):
                        j_elem_overlap.append(i)
                    else:
                        j_elem.relation[relation] = [i]
