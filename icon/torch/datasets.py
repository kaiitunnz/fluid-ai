import os
from PIL import Image  # type: ignore
from typing import Callable, List, Optional, Tuple

import torch
from torchvision import transforms  # type: ignore
from typing_extensions import Self

from ...torchutils.wrapper import DatasetWrapper
from .models import IconLabelerTransform


class RicoIconDataset(DatasetWrapper):
    """
    A wrapper for the RicoIcon dataset.

    Attributes
    ----------
    root : str
        Path to the root directory of the dataset.
    pretrained : bool
        Whether the model to run on use pre-trained weights.
    classes : List[str]
        List of output class names.
    icon_paths : List[Tuple[str, str]]
        List of tuples of the form `(icon_class, fname)` indicating the label and
        filename of each sample.
    directional_icons : List[str]
        List of icon classes considered directional icons.
    transform : Callable
        Transformation to be performed on non-directional icons.
    directional_transform : Callable
        Transformation to be performed on directional icons.
    augment : bool
        Whether to append the data augmentation steps to the end of each transformation.
    """

    root: str
    pretrained: bool
    classes: List[str]
    icon_paths: List[Tuple[str, str]]
    directional_icons: List[str]
    transform: Callable
    directional_transform: Callable
    augment: bool

    def __init__(
        self,
        root: str,
        pretrained: bool,
        transform: Callable,
        directional_transform: Callable,
        directional_icons: Optional[List[str]] = None,
        augment: bool = False,
    ):
        """
        Parameters
        ----------
        root : str
            Path to the root directory of the dataset.
        pretrained : bool
            Whether the model to run on use pre-trained weights.
        transform : Callable
            Transformation to be performed on non-directional icons.
        directional_transform : Callable
            Transformation to be performed on directional icons.
        directional_icons : Optional[List[str]]
            List of icon classes considered directional icons. `None` if there is
            no directional icon class.
        augment : bool
            Whether to append the data augmentation steps to the end of each transformation.
        """
        self.root = root
        self.pretrained = pretrained
        self.directional_icons = directional_icons or []

        self.classes: List[str] = []
        self.icon_paths: List[Tuple[str, str]] = []
        for icon_class in os.listdir(self.root):
            icon_class_dir = os.path.join(self.root, icon_class)
            if os.path.isdir(icon_class_dir):
                self.classes.append(icon_class)
                self.icon_paths.extend(
                    (icon_class, fname) for fname in os.listdir(icon_class_dir)
                )
        self.class2idx = {cls: i for i, cls in enumerate(self.classes)}

        if augment:
            self.transform = transforms.Compose(
                [
                    transform,
                    IconLabelerTransform.augment_transform(directional=False),
                ]
            )
            self.directional_transform = transforms.Compose(
                [
                    directional_transform,
                    IconLabelerTransform.augment_transform(directional=True),
                ]
            )
        else:
            self.transform = transform
            self.directional_transform = directional_transform

    def __len__(self):
        return len(self.icon_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        icon_class, icon_fname = self.icon_paths[idx]

        icon_path = os.path.join(self.root, icon_class, icon_fname)
        icon = Image.open(icon_path).convert("RGB")

        if icon_class in self.directional_icons:
            transformed = self.directional_transform(icon)
        else:
            transformed = self.transform(icon)

        return transformed, self.class2idx[icon_class]

    def labels(self) -> List[int]:
        return [self.class2idx[icon_class] for icon_class, _ in self.icon_paths]

    @classmethod
    def get_dataset_splits(
        cls,
        root_dir: str,
        pretrained: bool,
        train_transform: Callable,
        directional_transform: Callable,
        valid_transform: Callable,
        directional_icons: Optional[List[str]] = None,
        train_augment: bool = False,
    ) -> Tuple[Self, Self, Self, List[str]]:
        """Gets the training, validation, and test splits of the dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset within which there must be
            three subdirectories: "train", "val", and "test".
        pretrained : bool
            Whether the model to run on use pre-trained weights.
        train_transform : Callable
            Transformation to be performed on non-directional icons during training.
        directional_transform : Callable
            Transformation to be performed on directional icons during training.
        valid_transform : Callable
            Transformation for model validation or inference.
        directional_icons : Optional[List[str]]
            List of icon classes considered directional icons. `None` if there is
            no directional icon class.
        train_augment : bool
            Whether to append the data augmentation steps to the end of each transformation.

        Returns
        -------
        Tuple[RicoIconDataset, RicoIconDataset, RicoIconDataset, List[str]]
            `(train_split, val_split, test_split, class_names)`.
        """
        train = cls(
            os.path.join(root_dir, "train"),
            pretrained,
            train_transform,
            directional_transform,
            directional_icons,
            train_augment,
        )
        val = cls(
            os.path.join(root_dir, "val"),
            pretrained,
            valid_transform,
            directional_transform,
            directional_icons=None,
            augment=False,
        )
        test = cls(
            os.path.join(root_dir, "test"),
            pretrained,
            valid_transform,
            directional_transform,
            directional_icons=None,
            augment=False,
        )
        assert train.classes == val.classes and val.classes == test.classes
        return train, val, test, train.classes
