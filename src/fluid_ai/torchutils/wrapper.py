import functools
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore


class ModelWrapper(nn.Module):
    """
    A wrapper for PyTorch's Module.

    Attributes
    ----------
    model : Module
        Model to be wrapped.
    pretrained : bool
        Whether the model has pre-trained weights.
    classes : List[Any]
        List of class names with respect to the model's output indices.
    transform : Callable
        Transformation to preprocess the input to the model.
    """

    model: nn.Module
    pretrained: bool
    classes: List[Any]
    transform: Callable

    def __init__(
        self,
        model: nn.Module,
        pretrained: bool,
        classes: List[Any],
        transform: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        model : Module
            Model to be wrapped.
        pretrained : bool
            Whether the model has pre-trained weights.
        classes : List[Any]
            List of class names with respect to the model's output indices.
        transform : Optional[Callable]
            Transformation to preprocess the input to the model. None if not available.
        """
        super().__init__()
        for params in model.parameters():
            params.requires_grad = True
        self.model = model
        self.pretrained = pretrained
        self.classes = classes
        self.transform = transforms.Compose([]) if transform is None else transform

    def device(self) -> torch.device:
        """Returns the device on which the model resides

        Returns
        -------
        device
            Device on which the model resides.
        """
        return next(self.model.parameters()).device

    def forward(self, x: Any) -> Any:
        return self.model(x.to(self.device()))

    @abstractmethod
    def get_pred_idx(self, out: torch.Tensor) -> torch.Tensor:
        """Gets the class indices corresponding to the model's outputs

        Parameters
        ----------
        out : Tensor
            The model's outputs.

        Returns
        -------
        Tensor
            Class indices corresponding to the model's outputs.
        """
        raise NotImplementedError()

    def get_preds(self, out: Any) -> List[Any]:
        """Gets the class names corresponding to the model's outputs

        Parameters
        ----------
        out : Tensor
            The model's outputs.

        Returns
        -------
        Tensor
            Class names corresponding to the model's outputs.
        """
        pred_idx = self.get_pred_idx(out).to(torch.int).tolist()
        return [self.classes[i] for i in pred_idx]


class DatasetWrapper(Dataset):
    """
    A wrapper for PyTorch's Dataset.
    """

    def __len__(self):
        raise NotImplementedError()

    def labels(self) -> List[int]:
        """Gets the list of label indices of all data samples

        Returns
        -------
        List[int]
            List of label indices of all data samples.
        """
        raise NotImplementedError()
