from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore


class ModelWrapper(nn.Module):
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
        super().__init__()
        for params in model.parameters():
            params.requires_grad = True
        self.model = model
        self.pretrained = pretrained
        self.classes = classes
        self.transform = transforms.Compose([]) if transform is None else transform

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, x: Any) -> Any:
        return self.model(x.to(self.device()))


class DatasetWrapper(Dataset):
    def __len__(self):
        raise NotImplementedError()

    def labels(self) -> List[int]:
        raise NotImplementedError()
