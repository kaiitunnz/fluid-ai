import os
from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 224


def get_train_transform(
    pretrained: bool, image_size: int = IMAGE_SIZE
) -> transforms.Compose:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ToTensor(),
            normalize_transform(pretrained),
        ]
    )
    return train_transform


def get_valid_transform(
    pretrained: bool, image_size: int = IMAGE_SIZE
) -> transforms.Compose:
    valid_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize_transform(pretrained),
        ]
    )
    return valid_transform


def get_infer_transform(
    pretrained: bool, image_size: int = IMAGE_SIZE
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            normalize_transform(pretrained),
        ]
    )


def normalize_transform(pretrained: bool) -> transforms.Normalize:
    if pretrained:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return normalize


def get_datasets(
    root_dir: str, pretrained: bool, image_size: int = IMAGE_SIZE
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    train = ImageFolder(
        os.path.join(root_dir, "train"),
        transform=get_train_transform(pretrained, image_size),
    )
    val = ImageFolder(
        os.path.join(root_dir, "val"),
        transform=get_valid_transform(pretrained, image_size),
    )
    test = ImageFolder(
        os.path.join(root_dir, "test"),
        transform=get_valid_transform(pretrained, image_size),
    )
    return train, val, test, train.classes


def get_data_loaders(data: Dataset, batch_size: int = 16, num_workers: int = 4):
    return DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
