from typing import Any, List

import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    model: nn.Module
    pretrained: bool
    classes: List[Any]

    def __init__(self, model: nn.Module, pretrained: bool, classes: List[Any]):
        super().__init__()
        for params in model.parameters():
            params.requires_grad = True
        self.model = model
        self.pretrained = pretrained
        self.classes = classes

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, x: Any) -> Any:
        return self.model(x.to(self.device()))
