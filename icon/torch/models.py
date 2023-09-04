from typing import List

import torchvision.models as models
import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    model: nn.Module
    pretrained: bool
    classes: List[str]

    def __init__(self, model: nn.Module, pretrained: bool, classes: List[str]):
        super().__init__()
        self.model = model
        self.pretrained = pretrained
        self.classes = classes

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, x: ...) -> ...:
        return self.model(x.to(self.device()))


def build_efficientnet_v2(classes: List[str], pretrained: bool = True) -> ModelWrapper:
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.efficientnet_v2_s(models.EfficientNet_V2_S_Weights.DEFAULT)
    else:
        print("[INFO]: Not loading pre-trained weights")
        model = models.efficientnet_v2_s()
    for params in model.parameters():
        params.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=len(classes))  # type: ignore
    return ModelWrapper(model, pretrained, classes)


def build_resnet_50(classes: List[str], pretrained: bool = True) -> ModelWrapper:
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.resnet50(models.ResNet50_Weights.DEFAULT)
    else:
        print("[INFO]: Not loading pre-trained weights")
        model = models.resnet50()
    for params in model.parameters():
        params.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=len(classes))
    return ModelWrapper(model, pretrained, classes)
