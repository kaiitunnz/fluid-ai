from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch

_Value = Any


class Metric:
    """
    Model evaluation metric.

    Attributes
    ----------
    name : str
        Name of the metric.
    keephist : bool
        Whether to track the history of the metric values.
    """

    name: str
    keephist: bool
    _history: List[_Value]  # Tracks the history of metric values
    _value: Optional[_Value]  # Current metric value

    def __init__(self, name: str, keephist: bool = True):
        """
        Parameters
        ----------
        name : str
            Name of the metric.
        keephist : bool
            Whether to track the history of the metric values.
        """
        self.name = name
        self.keephist = keephist
        self._history = []
        self._value = None

    @abstractmethod
    def compute(self, preds: Any, labels: Any) -> _Value:
        """Computes the metric values given the model's predictions and ground-truth
        labels

        Parameters
        ----------
        preds : Any
            Model predictions.
        labels : Any
            Ground-truth labels.

        Returns
        -------
        Any
            Metric value.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Resets the metric computation"""
        raise NotImplementedError()

    @abstractmethod
    def is_better(self, old: _Value, new: _Value) -> bool:
        """Indicates whether `new` is better than `old`

        Parameters
        ----------
        old : Any
            Old metric value.
        new : Any
            New metric value.

        Returns
        -------
        bool
            Whether `new` is better than `old`.
        """
        raise NotImplementedError()

    def is_current_better(self, old: Optional[_Value]) -> bool:
        """Indicates whether the current metric value is better than `old`

        Parameters
        ----------
        old : Optional[Any]
            Old metric value

        Returns
        -------
        bool
            Whether the current metric value is better than `old`.
        """
        if old is None:
            if self._value is None:
                return False
            return True
        return self.is_better(old, self._value)

    def update(self, preds: Any, labels: Any):
        """Updates the current metric value given the model's predictions and ground-truth
        labels

        Parameters
        ----------
        preds : Any
            Model predictions.
        labels : Any
            Ground-truth labels.
        """
        self._value = self.compute(preds, labels)

    def value(self) -> _Value:
        """Returns the current value

        The metric value must have been computed at least once using the `Metric.update()`
        method.

        Returns
        -------
        Any
            Current metric value.
        """
        assert self._value is not None
        return self._value

    def history(self) -> List[_Value]:
        """Returns the history of the metric values

        Returns
        -------
        List[Any]
            History of the metric values.
        """
        return self._history

    def clear(self):
        """Clears the history and resets the metric computation"""
        self._history = []
        self._value = None
        self.reset()

    def commit(self) -> _Value:
        """Commits the updated metric value to the metric's history

        Must be called to update the history. Otherwise, the current value, computed
        with `Metric.update()`, will not be added to the history.

        If `keephist` is False, do nothing.

        Returns
        -------
        Any
            Current metric value.
        """
        assert self._value is not None
        if self.keephist:
            self._history.append(self._value)
        return self._value


class MetricList:
    """
    List of model evaluation metrics.

    Attributes
    ----------
    metrics : List[Metric]
        List of model evaluation metrics.
    name : str
        Name of the metric list.
    main_metric : Metric
        The representative metric.
    """

    metrics: List[Metric]
    name: str
    main_metric: Metric

    def __init__(
        self,
        metrics: List[Metric],
        name: str = "MetricList",
        main_metric: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        metrics : List[Metric]
            List of model evaluation metrics.
        name : str
            Name of the metric list.
        main_metric : Optional[str]
            Name of the representative metric. If `None`, the first metric in the
            list will be used.
        """
        assert len(metrics) > 0
        self.metrics = metrics
        self.name = name
        if main_metric is None:
            self.main_metric = metrics[0]
        else:
            for metric in metrics:
                if metric.name == main_metric:
                    self.main_metric = metric
                    return
            raise ValueError("Main metric not found!")

    def compute(self, preds: Any, labels: Any) -> List[_Value]:
        """Computes the metric values for all metrics given the model's predictions
        and ground-truth labels

        Parameters
        ----------
        preds : Any
            Model predictions.
        labels : Any
            Ground-truth labels.

        Returns
        -------
        Any
            List of metric values.

        """
        return [metric.compute(preds, labels) for metric in self.metrics]

    def reset(self):
        """Resets the metrics' computation"""
        for metric in self.metrics:
            metric.reset()

    def update(self, preds: Any, labels: Any):
        """Updates the current metric values of all the metrics given the model's
        predictions and ground-truth labels

        Parameters
        ----------
        preds : Any
            Model predictions.
        labels : Any
            Ground-truth labels.
        """
        for metric in self.metrics:
            metric.update(preds, labels)

    def main_value(self) -> _Value:
        """Returns the current value of the main metric

        Returns
        -------
        Any
            Current value of the representative metric
        """
        return self.main_metric.value()

    def is_current_better(self, old: Optional[_Value]) -> bool:
        """Indicates whether the current metric value of the main metric is better
        than `old`

        Parameters
        ----------
        old : Optional[Any]
            Old value of the main metric.

        Returns
        -------
        bool
            Whether the current metric value of the main metric is better than `old`.
        """
        return self.main_metric.is_current_better(old)

    def values(self) -> Dict[str, _Value]:
        """Returns the current values of all the metrics

        The metric values must have been computed at least once using the `MetricList.update()`
        method.

        Returns
        -------
        Dict[str, Any]
            A mapping from the metric names to the current metric values.
        """
        return {metric.name: metric.value() for metric in self.metrics}

    def value_list(self) -> List[_Value]:
        """Returns the list of the current values of all the metrics

        The metric values must have been computed at least once using the `MetricList.update()`
        method.

        Returns
        -------
        List[Any]
            List of the current values of all the metrics.
        """
        return [metric.value() for metric in self.metrics]

    def histories(self) -> Dict[str, List[_Value]]:
        """Returns the histories of all the metrics

        Returns
        -------
        Dict[str, List[_Value]]
            A mapping from the metric names to the metrics' histories.
        """
        return {metric.name: metric.history() for metric in self.metrics}

    def clear(self):
        """Clears the histories and resets the computation of all the metrics"""
        for metric in self.metrics:
            metric.clear()

    def commit(self) -> Dict[str, _Value]:
        """Commits the updated metric values to the metrics' histories

        Must be called to update the histories. Otherwise, the current values, computed
        with `MetricList.update()`, will not be added to the histories.

        If `keephist` is False, do nothing.

        Returns
        -------
        Any
            Current metric values.
        """
        return {metric.name: metric.commit() for metric in self.metrics}


class Accuracy(Metric):
    """
    Accuracy.

    Attributes
    ----------
    name : str
        Name of the metric.
    keephist : bool
        Whether to track the history of the metric values.
    """

    _correct: int
    _total: int

    def __init__(self, name: str, keephist: bool = True):
        super().__init__(name, keephist)
        self._correct = 0
        self._total = 0

    def compute(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        assert preds.size() == labels.size()
        self._correct += int((preds == labels).sum().item())
        self._total += labels.size()[0]
        if self._total == 0:
            return 0
        return self._correct / self._total

    def reset(self):
        self._correct = self._total = 0

    def is_better(self, old: float, new: float) -> bool:
        return new >= old


class BinaryF1Score(Metric):
    """
    Binary F1 score.

    Attributes
    ----------
    name : str
        Name of the metric.
    keephist : bool
        Whether to track the history of the metric values.
    """

    _tp: int
    _fp: int
    _fn: int
    _true: float

    def __init__(self, name: str, keephist: bool = True, inverted: bool = False):
        """
        If `inverted` the negative predictions/labels will be considered positive,
        and vice versa.

        Parameters
        ----------
        name : str
            Name of the metric.
        keephist : bool
            Whether to track the history of the metric values.
        inverted : bool
            Whether to invert the positive and negative labels.
        """
        super().__init__(name, keephist)
        self._tp = self._fp = self._fn = 0
        self._true = 0 if inverted else 1

    def compute(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        assert preds.size() == labels.size()
        self._tp += int(
            torch.logical_and((preds == self._true), (labels == self._true))
            .sum()
            .item()
        )
        self._fp += int(
            torch.logical_and((preds == self._true), (labels != self._true))
            .sum()
            .item()
        )
        self._fn += int(
            torch.logical_and((preds != self._true), (labels == self._true))
            .sum()
            .item()
        )
        denom = self._tp + ((self._fp + self._fn) / 2)
        if denom == 0:
            return 0.0
        return self._tp / denom

    def reset(self):
        self._tp = self._fp = self._fn = 0

    def is_better(self, old: float, new: float) -> bool:
        return new >= old


class BinaryPrecision(Metric):
    """
    Binary precision.

    Attributes
    ----------
    name : str
        Name of the metric.
    keephist : bool
        Whether to track the history of the metric values.
    """

    _tp: int
    _fp: int
    _true: float

    def __init__(self, name: str, keephist: bool = True, inverted: bool = False):
        """
        If `inverted` the negative predictions/labels will be considered positive,
        and vice versa.

        Parameters
        ----------
        name : str
            Name of the metric.
        keephist : bool
            Whether to track the history of the metric values.
        inverted : bool
            Whether to invert the positive and negative labels.
        """
        super().__init__(name, keephist)
        self._tp = self._fp = 0
        self._true = 0 if inverted else 1

    def compute(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        assert preds.size() == labels.size()
        self._tp += int(
            torch.logical_and((preds == self._true), (labels == self._true))
            .sum()
            .item()
        )
        self._fp += int(
            torch.logical_and((preds == self._true), (labels != self._true))
            .sum()
            .item()
        )
        denom = self._tp + self._fp
        if denom == 0:
            return 0.0
        return self._tp / denom

    def reset(self):
        self._tp = self._fp = 0

    def is_better(self, old: float, new: float) -> bool:
        return new >= old


class BinaryRecall(Metric):
    """
    Binary recall.

    Attributes
    ----------
    name : str
        Name of the metric.
    keephist : bool
        Whether to track the history of the metric values.
    """

    _tp: int
    _fn: int
    _true: float

    def __init__(self, name: str, keephist: bool = True, inverted: bool = False):
        """
        If `inverted` the negative predictions/labels will be considered positive,
        and vice versa.

        Parameters
        ----------
        name : str
            Name of the metric.
        keephist : bool
            Whether to track the history of the metric values.
        inverted : bool
            Whether to invert the positive and negative labels.
        """
        super().__init__(name, keephist)
        self._tp = self._fp = 0
        self._true = 0 if inverted else 1

    def compute(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        assert preds.size() == labels.size()
        self._tp += int(
            torch.logical_and((preds == self._true), (labels == self._true))
            .sum()
            .item()
        )
        self._fn += int(
            torch.logical_and((preds != self._true), (labels == self._true))
            .sum()
            .item()
        )
        denom = self._tp + self._fn
        if denom == 0:
            return 0.0
        return self._tp / (self._tp + self._fn)

    def reset(self):
        self._tp = self._fn = 0

    def is_better(self, old: float, new: float) -> bool:
        return new >= old
