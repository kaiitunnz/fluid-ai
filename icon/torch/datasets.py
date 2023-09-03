import os
from PIL import Image
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE = 224


class RicoIconDataset(Dataset):
    root: str
    training: bool
    pretrained: bool
    image_size: int
    classes: List[str]
    icon_paths: List[Tuple[str, str]]
    directional_icons: Optional[List[str]]
    transform: Callable
    directional_transform: Callable

    def __init__(
        self,
        root: str,
        training: bool,
        pretrained: bool,
        image_size: int = IMAGE_SIZE,
        directional_icons: Optional[List[str]] = None,
    ):
        self.root = root
        self.training = training
        self.pretrained = pretrained
        self.image_size = image_size
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

        if self.training:
            self.transform = self._train_transform()
            self.directional_transform = self._train_directional_transform()
        else:
            self.transform = self._valid_transform()
            self.directional_transform = self.transform

    def __len__(self):
        return len(self.icon_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        icon_class, icon_fname = self.icon_paths[idx]

        icon_path = os.path.join(self.root, icon_class, icon_fname)
        icon = Image.open(icon_path).convert("RGB")

        if icon_class in self.directional_icons:
            transformed = self.directional_transform(icon)
        else:
            transformed = self.transform(icon)

        return transformed, self.class2idx[icon_class]

    def _train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.ToTensor(),
                _normalize_transform(self.pretrained),
            ]
        )

    def _train_directional_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.ToTensor(),
                _normalize_transform(self.pretrained),
            ]
        )

    def _valid_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                _normalize_transform(self.pretrained),
            ]
        )


def get_infer_transform(
    pretrained: bool, image_size: int = IMAGE_SIZE
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            _normalize_transform(pretrained),
        ]
    )


def _normalize_transform(pretrained: bool) -> transforms.Normalize:
    if pretrained:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return normalize


def get_datasets(
    root_dir: str,
    pretrained: bool,
    directional_icons: Optional[List[str]] = None,
    image_size: int = IMAGE_SIZE,
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    train = RicoIconDataset(
        os.path.join(root_dir, "train"),
        True,
        pretrained,
        image_size,
        directional_icons,
    )
    val = RicoIconDataset(
        os.path.join(root_dir, "val"),
        False,
        pretrained,
        image_size,
    )
    test = RicoIconDataset(
        os.path.join(root_dir, "test"),
        False,
        pretrained,
        image_size,
    )
    assert train.classes == val.classes and val.classes == test.classes
    return train, val, test, train.classes


def get_data_loaders(data: Dataset, batch_size: int = 16, num_workers: int = 4):
    return DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
