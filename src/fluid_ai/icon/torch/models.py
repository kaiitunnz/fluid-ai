from typing import List, Tuple, Union

import torchvision.models as models  # type: ignore
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F  # type: ignore

from ...torchutils import ModelWrapper

DEFAULT_IMAGE_SIZE = 224


class IconLabelerTransform(nn.Module):
    """
    Transformation for classification-based icon-labeling models.

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
        resize_size: Union[Tuple[int, int], int] = DEFAULT_IMAGE_SIZE,
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

    def train_transform(self, directional: bool = False) -> transforms.Compose:
        """Gets default image transformation with data augmentation for training a
        model

        Parameters
        ----------
        directional : bool
            Whether to include only data augmentation steps for directional icon elements.

        Returns
        -------
        transforms.Compose
            Image transformation with data augmentation for training a model.
        """
        return transforms.Compose(
            [
                transforms.Resize(self.resize_size, self.interpolation),
                IconLabelerTransform.augment_transform(directional),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    @staticmethod
    def augment_transform(directional: bool) -> transforms.Compose:
        """Gets default data augmentation transformation

        Parameters
        ----------
        directional : bool
            Whether to include only data augmentation steps for directional icon elements.

        Returns
        -------
        transforms.Compose
            Data augmentation transformation for training a model.
        """
        if directional:
            return transforms.Compose(
                [
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                ]
            )
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            ]
        )

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


class IconLabelerWrapper(ModelWrapper):
    """
    A wrapper for an icon labeling model.

    Attributes
    ----------
    transform : IconLabelerTransform
        Input image transformation for inference.
    """

    transform: IconLabelerTransform

    def get_pred_idx(self, out: torch.Tensor) -> torch.Tensor:
        return torch.max(out.data, 1)[1]


def build_efficientnet_v2(
    classes: List[str], pretrained: bool = True
) -> IconLabelerWrapper:
    """Builds an icon labeling model with an EfficientNetV2-S backbone.

    Parameters
    ----------
    classes : List[str]
        Names of the output classes.
    pretrained : bool
        Whether to use pre-trained weights.

    Returns
    -------
    IconLabelerWrapper
        A wrapped icon labeling model.
    """
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
    transform = IconLabelerTransform()
    return IconLabelerWrapper(model, pretrained, classes, transform)


def build_resnet_50(classes: List[str], pretrained: bool = True) -> IconLabelerWrapper:
    """Builds an icon labeling model with a ResNet-50 backbone

    Parameters
    ----------
    classes : List[str]
        Names of the output classes.
    pretrained : bool
        Whether to use pre-trained weights.

    Returns
    -------
    IconLabelerWrapper
        A wrapped icon labeling model.
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
    model.fc = nn.Linear(in_features=in_features, out_features=len(classes))
    transform = IconLabelerTransform()
    return IconLabelerWrapper(model, pretrained, classes, transform)
