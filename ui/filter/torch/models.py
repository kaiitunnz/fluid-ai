from typing import Any, Callable, List, Optional

import torchvision.models as models  # type: ignore
import torch
import torch.nn as nn

from ....torchutils import ModelWrapper


class FilterModelWrapper(ModelWrapper):
    threshold: float

    def __init__(
        self,
        model: nn.Module,
        pretrained: bool,
        classes: List[Any],
        transform: Optional[Callable] = None,
        threshold: float = 0.5,
    ):
        super().__init__(model, pretrained, classes, transform)
        self.threshold = threshold

    def get_pred_idx(self, out: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(out)
        result[out > self.threshold] = 1
        return result


def build_efficientnet_v2_s(
    classes: List[Any],
    masked: bool = False,
    pretrained: bool = True,
    imgsize: Optional[int] = None,
) -> ModelWrapper:
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.efficientnet_v2_s(models.EfficientNet_V2_S_Weights.DEFAULT)
    else:
        print("[INFO]: Not loading pre-trained weights")
        model = models.efficientnet_v2_s()

    if masked:
        # Add a mask channel.
        pretrained_conv2d: torch.nn.Conv2d = model.features[0][0]  # type: ignore
        new_first_conv2d = torch.nn.Conv2d(
            pretrained_conv2d.in_channels + 1,
            pretrained_conv2d.out_channels,
            pretrained_conv2d.kernel_size,  # type: ignore
            pretrained_conv2d.stride,  # type: ignore
            pretrained_conv2d.padding,  # type: ignore
            pretrained_conv2d.dilation,  # type: ignore
            pretrained_conv2d.groups,
            pretrained_conv2d.bias is not None,
            pretrained_conv2d.padding_mode,
        )
        new_first_conv2d.weight[:, :3, :, :].data = pretrained_conv2d.weight
        model.features[0][0] = new_first_conv2d

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=1)  # type: ignore
    if imgsize is None:
        transforms = models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
    else:
        transforms = models.EfficientNet_V2_S_Weights.DEFAULT.transforms(
            resize_size=imgsize, crop_size=imgsize
        )

    return FilterModelWrapper(model, pretrained, classes, transforms)


def build_efficientnet_v2_m(
    classes: List[Any],
    masked: bool = False,
    pretrained: bool = True,
    imgsize: Optional[int] = None,
) -> ModelWrapper:
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.efficientnet_v2_m(models.EfficientNet_V2_M_Weights.DEFAULT)
    else:
        print("[INFO]: Not loading pre-trained weights")
        model = models.efficientnet_v2_m()

    if masked:
        # Add a mask channel.
        pretrained_conv2d: torch.nn.Conv2d = model.features[0][0]  # type: ignore
        new_first_conv2d = torch.nn.Conv2d(
            pretrained_conv2d.in_channels + 1,
            pretrained_conv2d.out_channels,
            pretrained_conv2d.kernel_size,  # type: ignore
            pretrained_conv2d.stride,  # type: ignore
            pretrained_conv2d.padding,  # type: ignore
            pretrained_conv2d.dilation,  # type: ignore
            pretrained_conv2d.groups,
            pretrained_conv2d.bias is not None,
            pretrained_conv2d.padding_mode,
        )
        new_first_conv2d.weight[:, :3, :, :].data = pretrained_conv2d.weight
        model.features[0][0] = new_first_conv2d

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=1)  # type: ignore
    if imgsize is None:
        transforms = models.EfficientNet_V2_M_Weights.DEFAULT.transforms()
    else:
        transforms = models.EfficientNet_V2_M_Weights.DEFAULT.transforms(
            resize_size=imgsize, crop_size=imgsize
        )

    return FilterModelWrapper(
        model,
        pretrained,
        classes,
        transforms,
    )


def build_resnet_50(
    classes: List[Any],
    pretrained: bool = True,
    imgsize: Optional[int] = None,
) -> ModelWrapper:
    if pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.resnet50(models.ResNet50_Weights.DEFAULT)
    else:
        print("[INFO]: Not loading pre-trained weights")
        model = models.resnet50()
    for params in model.parameters():
        params.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=1)
    if imgsize is None:
        transforms = models.ResNet50_Weights.DEFAULT.transforms()
    else:
        transforms = models.ResNet50_Weights.DEFAULT.transforms(
            resize_size=imgsize, crop_size=imgsize
        )
    return FilterModelWrapper(model, pretrained, classes, transforms)
