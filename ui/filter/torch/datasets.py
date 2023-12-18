import csv
import os
from PIL import Image  # type: ignore
from typing import Callable, List, NamedTuple, Optional, Tuple
from typing_extensions import Self

import torch
import yaml  # type: ignore
from cachetools import LRUCache  # type: ignore
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms  # type: ignore

from ....base import BBox
from ....torchutils.wrapper import DatasetWrapper


class _BoundaryEntry(NamedTuple):
    screenshot_id: str
    bbox: BBox
    label: int


class RicoValidDataset(DatasetWrapper):
    VALID: str = "valid"
    INVALID: str = "invalid"

    training: bool
    initial_transform: Callable
    final_transform: Callable

    def __init__(self, training: bool, initial_transform: Callable):
        self.training = training
        self.initial_transform = initial_transform

        if self.training:
            self.final_transform = self._train_transform()
        else:
            self.final_transform = self._valid_transform()

    def _train_transform(self) -> Callable:
        return transforms.RandomHorizontalFlip(p=0.5)

    def _valid_transform(self) -> Callable:
        return transforms.Compose([])


class RicoValidElement(RicoValidDataset):
    root: str
    sample_paths: List[Tuple[str, str]]
    transform: Callable

    def __init__(self, root: str, training: bool, initial_transform: Callable):
        super().__init__(training, initial_transform)

        self.root = root

        self.sample_paths: List[Tuple[str, str]] = []
        valid_dir = os.path.join(self.root, self.VALID)
        invalid_dir = os.path.join(self.root, self.INVALID)
        self.sample_paths.extend((self.VALID, fname) for fname in os.listdir(valid_dir))
        self.sample_paths.extend(
            (self.INVALID, fname) for fname in os.listdir(invalid_dir)
        )

        self.transform = transforms.Compose(
            [self.initial_transform, self.final_transform]
        )

    def labels(self) -> List[int]:
        return [(1 if label == self.VALID else 0) for label, _ in self.sample_paths]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cls, fname = self.sample_paths[idx]

        fpath = os.path.join(self.root, cls, fname)
        element = Image.open(fpath).convert("RGB")
        transformed = self.transform(element)

        return transformed, (1 if cls == self.VALID else 0)

    @classmethod
    def get_dataset_splits(
        cls,
        root: str,
        initial_transform: Callable,
    ) -> Tuple[Self, Self, Self]:
        train = cls(
            os.path.join(root, "train"),
            True,
            initial_transform,
        )
        val = cls(
            os.path.join(root, "val"),
            True,
            initial_transform,
        )
        test = cls(
            os.path.join(root, "test"),
            True,
            initial_transform,
        )

        return train, val, test


class RicoValidBoundary(RicoValidDataset):
    screenshot_dir: str
    data_path: str
    sample_entries: List[_BoundaryEntry]
    screenshot_cache: LRUCache[str, Image.Image]

    def __init__(
        self,
        screenshot_dir: str,
        data_path: str,
        training: bool,
        initial_transform: Callable,
        cache_size: int = 0,
    ):
        super().__init__(training, initial_transform)

        self.screenshot_dir = screenshot_dir
        self.data_path = data_path

        self.sample_entries = []
        with open(self.data_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                screenshot_id, *str_bbox, label = row
                x0, y0, x1, y1 = map(float, str_bbox)
                self.sample_entries.append(
                    _BoundaryEntry(screenshot_id, BBox((x0, y0), (x1, y1)), int(label))
                )

        self.screenshot_cache = LRUCache(cache_size)

    def labels(self) -> List[int]:
        return [entry.label for entry in self.sample_entries]

    def __len__(self):
        return len(self.sample_entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        screenshot_id, bbox, label = self.sample_entries[idx]

        screenshot = self.screenshot_cache.get(screenshot_id)
        if screenshot is None:
            screenshot_path = os.path.join(self.screenshot_dir, f"{screenshot_id}.jpg")
            screenshot = Image.open(screenshot_path).convert("RGB")
            self.screenshot_cache[screenshot_id] = screenshot
        initial_transformed: torch.Tensor = self.initial_transform(screenshot)
        mask = RicoValidBoundary._get_mask(initial_transformed, bbox)
        transformed = torch.concat([initial_transformed, mask], dim=0)

        return self.final_transform(transformed), label

    @staticmethod
    def _get_mask(
        image: torch.Tensor, bbox: BBox, normalized: bool = True
    ) -> torch.Tensor:
        if normalized:
            x0, y0, x1, y1 = bbox.scale(*image.size()[1:]).to_int_flattened()
        else:
            x0, y0, x1, y1 = bbox.to_int_flattened()
        mask = torch.zeros((1, *image.size()[1:]))
        mask[0, y0:y1, x0:x1] = 1.0
        return mask

    @classmethod
    def get_dataset_splits(
        cls,
        screenshot_dir: str,
        data_dir: str,
        initial_transform: Callable,
        cache_size: int = 1024,
    ) -> Tuple[Self, Self, Self]:
        train = cls(
            screenshot_dir,
            os.path.join(data_dir, "train.csv"),
            True,
            initial_transform,
            cache_size,
        )
        val = cls(
            screenshot_dir,
            os.path.join(data_dir, "val.csv"),
            False,
            initial_transform,
            cache_size,
        )
        test = cls(
            screenshot_dir,
            os.path.join(data_dir, "test.csv"),
            False,
            initial_transform,
            cache_size,
        )

        return train, val, test

    @classmethod
    def from_config(
        cls,
        config: str,
        initial_transform: Callable,
        cache_size: int = 1024,
    ) -> Tuple[Self, Self, Self]:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f.read())
        screenshot_dir = cfg["images"]
        train = cls(
            screenshot_dir,
            cfg["train"],
            True,
            initial_transform,
            cache_size,
        )
        val = cls(
            screenshot_dir,
            cfg["val"],
            False,
            initial_transform,
            cache_size,
        )
        test = cls(
            screenshot_dir,
            cfg["test"],
            False,
            initial_transform,
            cache_size,
        )

        return train, val, test


def get_data_loaders(
    data: Dataset,
    batch_size: int = 16,
    sampler: Optional[Sampler] = None,
    num_workers: int = 4,
):
    if sampler is None:
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    return DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )
