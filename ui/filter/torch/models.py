from typing import Any, Callable, List, Optional, Tuple, Union

import torchvision.models as models  # type: ignore
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F  # type: ignore

from ....torchutils import ModelWrapper


DEFAULT_EFFICIENTNETV2_S_IMGSIZE = 384
DEFAULT_EFFICIENTNETV2_M_IMGSIZE = 480
DEFAULT_RESNET50_IMGSIZE = 224


class FilterModelWrapper(ModelWrapper):
    """
    A wrapper for a UI filter model, aka an invalid UI detection model.

    Attributes
    ----------
    threshold: float
        A threshold value above which the input is considered valid.
    """

    threshold: float

    def __init__(
        self,
        model: nn.Module,
        pretrained: bool,
        classes: List[Any],
        transform: Optional[Callable] = None,
        threshold: float = 0.0,
    ):
        """
        Parameters
        ----------
        model : Module
            A PyTorch binary classification model.
        pretrained : bool
            Whether the model has pre-trained weights.
        classes : List[Any]
            Names of the output classes.
        transform : Optional[Callable]
            A Callable which transforms an image into a tensor.
        threshold: float
            A threshold value above which the input is considered valid.
        """
        super().__init__(model, pretrained, classes, transform)
        self.threshold = threshold

    def get_pred_idx(self, out: torch.Tensor) -> torch.Tensor:
        """Computes prediction labels based on the model's output

        Parameters
        ----------
        out : Tensor
            The model's output.

        Returns
        -------
        Tensor
            Output prediction labels.
        """
        result = torch.zeros_like(out)
        result[out > self.threshold] = 1
        return result


class UiFilterTransform(nn.Module):
    """
    Transformation for UI filter models

    Attributes
    ----------
    resize_size: List[int]
        The size to which the input images will be resized.
    mean: List[float]
        The channel means to which the images' pixel values will be normalized.
    std: List[float]
        The channel standard deviations to which the images' pixel values will
        be normalized.
    interpolation: InterpolationMode
        Interpolation model to be used for resizing the input images.
    """

    resize_size: List[int]
    mean: List[float]
    std: List[float]
    interpolation: transforms.InterpolationMode

    def __init__(
        self,
        *,
        resize_size: Union[Tuple[int, int], int] = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    ):
        """
        Parameters
        ----------
        resize_size: Union[Tuple[int, int], int]
            The size to which the input images will be resized. If it is a tuple,
            the target size will be exactly as specified. If it is an integer, the
            target size will be a square whose sides have the specified length.
        mean: Tuple[float, ...]
            The channel means to which the images' pixel values will be normalized.
        std: Tuple[float, ...]
            The channel standard deviations to which the images' pixel values will
            be normalized.
        interpolation: InterpolationMode
            Interpolation model to be used for resizing the input images.
        """
        super().__init__()
        self.resize_size = (
            list(resize_size)
            if isinstance(resize_size, tuple)
            else [resize_size, resize_size]
        )
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation)
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string


def build_efficientnet_v2_s(
    classes: List[Any],
    masked: bool = False,
    pretrained: bool = True,
    imgsize: int = DEFAULT_EFFICIENTNETV2_S_IMGSIZE,
) -> FilterModelWrapper:
    """Builds a UI filter model with an EfficientNetV2-S backbone

    Parameters
    ----------
    classes : List[Any]
        Names of the output classes.
    masked : bool
        Whether to use a mask channel (for a boundary-based model).
    pretrained : bool
        Whether to use pre-trained weights.
    imgsize : int
        Image size to be used.

    Returns
    -------
    FilterModelWrapper
        A wrapped UI filter model.
    """
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
    transforms = UiFilterTransform(resize_size=imgsize)
    return FilterModelWrapper(model, pretrained, classes, transforms)


def build_efficientnet_v2_m(
    classes: List[Any],
    masked: bool = False,
    pretrained: bool = True,
    imgsize: int = DEFAULT_EFFICIENTNETV2_M_IMGSIZE,
) -> FilterModelWrapper:
    """Builds a UI filter model with an EfficientNetV2-M backbone

    Parameters
    ----------
    classes : List[Any]
        Names of the output classes.
    masked : bool
        Whether to use a mask channel (for a boundary-based model).
    pretrained : bool
        Whether to use pre-trained weights.
    imgsize : int
        Image size to be used.

    Returns
    -------
    FilterModelWrapper
        A wrapped UI filter model.
    """
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
    transforms = UiFilterTransform(resize_size=imgsize)
    return FilterModelWrapper(model, pretrained, classes, transforms)


def build_resnet_50(
    classes: List[Any],
    pretrained: bool = True,
    imgsize: int = DEFAULT_RESNET50_IMGSIZE,
) -> FilterModelWrapper:
    """Builds a UI filter model with a ResNet-50 backbone

    Parameters
    ----------
    classes : List[Any]
        Names of the output classes.
    pretrained : bool
        Whether to use pre-trained weights.
    imgsize : int
        Image size to be used.

    Returns
    -------
    FilterModelWrapper
        A wrapped UI filter model.
    """
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
    transforms = UiFilterTransform(resize_size=imgsize)
    return FilterModelWrapper(model, pretrained, classes, transforms)
