import csv
import os
from PIL import Image  # type: ignore
from typing import Callable, List, NamedTuple, Tuple
from typing_extensions import Self

import torch
import yaml  # type: ignore
from cachetools import LRUCache  # type: ignore
from torchvision import transforms  # type: ignore

from ....base import NormalizedBBox, array_get_size
from ....torchutils.wrapper import DatasetWrapper


class _BoundaryEntry(NamedTuple):
    screenshot_id: str
    bbox: NormalizedBBox
    label: int


class RicoValidDataset(DatasetWrapper):
    """
    A base class of the RicoValid dataset.

    Attributes
    ----------
    VALID : str
        The string representation of the valid class.
    INVALID : str
        The string representation of the invalid class.
    training : bool
        Whether the instance is a training split.
    initial_transform : Callable
        The transformation to be called before any transformation specific to the dataset.
    final_transform : Callable
        The transformation to be callsed after any transformation specific to the dataset.
    """

    VALID: str = "valid"
    INVALID: str = "invalid"

    training: bool
    initial_transform: Callable
    final_transform: Callable

    def __init__(self, training: bool, initial_transform: Callable):
        """
        Parameters
        ----------
        training : bool
            Whether the instance is a training split.
        initial_transform : Callable
            The transformation to be called before any transformation specific to the dataset.
        """
        self.training = training
        self.initial_transform = initial_transform

        if self.training:
            self.final_transform = self._train_transform()
        else:
            self.final_transform = self._valid_transform()

    def _train_transform(self) -> Callable:
        """Returns a transformation to be used as `final_transform` for training.

        Returns
        -------
            The transformation to be used as `final_transform` for training.
        """
        return transforms.RandomHorizontalFlip(p=0.5)

    def _valid_transform(self) -> Callable:
        """Returns a transformation to be used as `final_transform` for validation
        or inference.

        Returns
        -------
            The transformation to be used as `final_transform` for validation
            or inference.
        """
        return transforms.Compose([])


class RicoValidElement(RicoValidDataset):
    """
    The RicoValid-elements dataset.

    Attributes
    ----------
    VALID : str
        The string representation of the valid class.
    INVALID : str
        The string representation of the invalid class.
    training : bool
        Whether the instance is a training split.
    initial_transform : Callable
        The transformation to be called before any transformation specific to the dataset.
    final_transform : Callable
        The transformation to be callsed after any transformation specific to the dataset.
    root : str
        Root directory of the dataset.
    sample_paths : List[Tuple[str, str]]
        List of paths to samples in the dataset.
    transform : Callable
        The transformation to be called on each sample image.
    """

    root: str
    sample_paths: List[Tuple[str, str]]
    transform: Callable

    def __init__(self, root: str, training: bool, initial_transform: Callable):
        """
        Parameters
        ----------
        root : str
            Root directory of the dataset.
        training : bool
            Whether the instance is a training split.
        initial_transform : Callable
            The transformation to be called before any transformation specific to the dataset.
        """
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
        """Returns a list of labels corresponding to the samples in `sample_paths`.

        Returns
        -------
        List[int]
            List of labels corresponding to the samples in `sample_paths`.
        """
        return [(1 if label == self.VALID else 0) for label, _ in self.sample_paths]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        cls, fname = self.sample_paths[idx]

        fpath = os.path.join(self.root, cls, fname)
        element = Image.open(fpath).convert("RGB")
        transformed = self.transform(element)

        return transformed, (1.0 if cls == self.VALID else 0.0)

    @classmethod
    def get_dataset_splits(
        cls,
        root: str,
        initial_transform: Callable,
    ) -> Tuple[Self, Self, Self]:
        """Gets the training, validation, and test splits of the dataset.

        Parameters
        ----------
        root : str
            Path to the root directory of the dataset within which there must be
            three subdirectories: "train", "val", and "test".
        initial_transform : Callable
            The transformation to be called before any transformation specific to the dataset.

        Returns
        -------
        Tuple[RicoValidElement, RicoValidElement, RicoValidElement]
            (train_split, val_split, test_split).
        """
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
    """
    The RicoValid-boundaries dataset.

    Attributes
    ----------
    VALID : str
        The string representation of the valid class.
    INVALID : str
        The string representation of the invalid class.
    training : bool
        Whether the instance is a training split.
    initial_transform : Callable
        The transformation to be called before any transformation specific to the dataset.
    final_transform : Callable
        The transformation to be callsed after any transformation specific to the dataset.
    screenshot_dir : str
        The directory of original screenshots.
    data_path : str
        The path to the dataset (a CSV file).
    sample_entries: List[_BoundaryEntry]
        The list of entries in the dataset.
    screenshot_cache: LRUCache[str, Image.Image]
        The cache of screenshot images for performance.
    """

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
        """
        Parameters
        ----------
        screenshot_dir : str
            The directory of original screenshots.
        data_path : str
            The path to the dataset (a CSV file).
        training : bool
            Whether the instance is a training split.
        initial_transform : Callable
            The transformation to be called before any transformation specific to the dataset.
        cache_size : int
            Size of the screenshot cache to be used.
        """
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
                    _BoundaryEntry(
                        screenshot_id,
                        NormalizedBBox.new((x0, y0), (x1, y1)),
                        int(label),
                    )
                )

        self.screenshot_cache = LRUCache(cache_size)

    def labels(self) -> List[int]:
        return [entry.label for entry in self.sample_entries]

    def __len__(self):
        return len(self.sample_entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        screenshot_id, bbox, label = self.sample_entries[idx]

        screenshot = self.screenshot_cache.get(screenshot_id)
        if screenshot is None:
            screenshot_path = os.path.join(self.screenshot_dir, f"{screenshot_id}.jpg")
            screenshot = Image.open(screenshot_path).convert("RGB")
            self.screenshot_cache[screenshot_id] = screenshot
        initial_transformed: torch.Tensor = self.initial_transform(screenshot)
        masked = RicoValidBoundary.mask(initial_transformed, bbox)

        return self.final_transform(masked), float(label)

    @staticmethod
    def mask(image: torch.Tensor, bbox: NormalizedBBox) -> torch.Tensor:
        """Returns a masked image tensor of shape 4 x H x W

        Parameters
        ----------
        image : Tensor
            A tensor of shape 3 x H x W.
        bbox : NormalizedBBox
            The bounding box of a UI element.

        Returns
        -------
        Tensor
            A masked image tensor of shape 4 x H x W
        """
        w, h = array_get_size(image)
        x0, y0, x1, y1 = bbox.to_bbox(w, h).to_int_flattened()
        mask = torch.zeros((1, w, h))
        mask[0, y0:y1, x0:x1] = 1.0
        return torch.concat([image, mask], dim=0)

    @classmethod
    def get_dataset_splits(
        cls,
        screenshot_dir: str,
        data_dir: str,
        initial_transform: Callable,
        cache_size: int = 1024,
    ) -> Tuple[Self, Self, Self]:
        """Gets the training, validation, and test splits of the dataset.

        Parameters
        ----------
        screenshot_dir: str
            The directory of original screenshots.
        data_dir: str
            Path to the dataset directory within which there must be three subdirectories:
            "train.csv", "val.csv", and "test.csv".
        initial_transform : Callable
            The transformation to be called before any transformation specific to the dataset.
        cache_size : int
            Size of the screenshot cache to be used.

        Returns
        -------
        Tuple[RicoValidBoundary, RicoValidBoundary, RicoValidBoundary]
            (train_split, val_split, test_split).
        """
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
        """Gets the training, validation, and test splits of the dataset from a config file.

        Parameters
        ----------
        config: str
            Path to the config file
        initial_transform : Callable
            The transformation to be called before any transformation specific to the dataset.
        cache_size : int
            Size of the screenshot cache to be used.

        Returns
        -------
        Tuple[RicoValidBoundary, RicoValidBoundary, RicoValidBoundary]
            (train_split, val_split, test_split).
        """
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
