from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch

_Value = Any


class Metric:
    name: str
    keephist: bool
    _history: List[_Value] = []
    _value: Optional[_Value] = None

    def __init__(self, name: str, keephist: bool = True):
        self.name = name
        self.keephist = keephist

    @abstractmethod
    def compute(self, preds: Any, labels: Any) -> _Value:
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def is_better(self, old: _Value, new: _Value) -> bool:
        raise NotImplementedError()

    def is_current_better(self, old: Optional[_Value]) -> bool:
        if old is None:
            if self._value is None:
                return False
            return True
        return self.is_better(old, self._value)

    def update(self, preds: Any, labels: Any):
        self._value = self.compute(preds, labels)

    def value(self) -> _Value:
        assert self._value is not None
        return self._value

    def history(self) -> List[_Value]:
        return self._history

    def clear(self):
        self._history = []
        self._value = None
        self.reset()

    def commit(self) -> _Value:
        assert self._value is not None
        if self.keephist:
            self._history.append(self._value)
        return self._value


class MetricList:
    metrics: List[Metric]
    name: str
    main_metric: Metric

    def __init__(
        self,
        metrics: List[Metric],
        name: str = "MetricList",
        main_metric: Optional[str] = None,
    ):
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
        return [metric.compute(preds, labels) for metric in self.metrics]

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, preds: Any, labels: Any):
        for metric in self.metrics:
            metric.update(preds, labels)

    def main_value(self) -> _Value:
        return self.main_metric.value()

    def is_current_better(self, old: Optional[_Value]) -> bool:
        return self.main_metric.is_current_better(old)

    def values(self) -> Dict[str, _Value]:
        return {metric.name: metric.value() for metric in self.metrics}

    def value_list(self) -> List[_Value]:
        return [metric.value() for metric in self.metrics]

    def histories(self) -> Dict[str, List[_Value]]:
        return {metric.name: metric.history() for metric in self.metrics}

    def clear(self):
        for metric in self.metrics:
            metric.clear()

    def commit(self) -> Dict[str, _Value]:
        return {metric.name: metric.commit() for metric in self.metrics}


class BinaryAccuracy(Metric):
    _correct: int = 0
    _total: int = 0

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
    _tp: int = 0
    _fp: int = 0
    _fn: int = 0

    def compute(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        assert preds.size() == labels.size()
        self._tp += int(torch.logical_and((preds == 1), (labels == 1)).sum().item())
        self._fp += int(torch.logical_and((preds == 1), (labels != 1)).sum().item())
        self._fn += int(torch.logical_and((preds != 1), (labels == 1)).sum().item())
        denom = self._tp + ((self._fp + self._fn) / 2)
        if denom == 0:
            return 0
        return self._tp / denom

    def reset(self):
        self._tp = self._fp = self._fn = 0

    def is_better(self, old: float, new: float) -> bool:
        return new >= old


class BinaryPrecision(Metric):
    _tp: int = 0
    _fp: int = 0

    def compute(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        assert preds.size() == labels.size()
        self._tp += int(torch.logical_and((preds == 1), (labels == 1)).sum().item())
        self._fp += int(torch.logical_and((preds == 1), (labels != 1)).sum().item())
        denom = self._tp + self._fp
        if denom == 0:
            return 0
        return self._tp / denom

    def reset(self):
        self._tp = self._fp = 0

    def is_better(self, old: float, new: float) -> bool:
        return new >= old


class BinaryRecall(Metric):
    _tp: int = 0
    _fn: int = 0

    def compute(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        assert preds.size() == labels.size()
        self._tp += int(torch.logical_and((preds == 1), (labels == 1)).sum().item())
        self._fn += int(torch.logical_and((preds != 1), (labels == 1)).sum().item())
        denom = self._tp + self._fn
        if denom == 0:
            return 0
        return self._tp / (self._tp + self._fn)

    def reset(self):
        self._tp = self._fn = 0

    def is_better(self, old: float, new: float) -> bool:
        return new >= old
