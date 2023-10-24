from abc import abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torchvision import transforms  # type: ignore

from ..base import UiElement
from .labeldroid import utils as labeldroid_utils
from .labeldroid.args import LabelDroidArgs
from .labeldroid.models.combined_model import LabelDroid
from .torch.datasets import get_infer_transform
from .torch.models import ModelWrapper
from .torch.utils import load_model


class BaseIconLabeller:
    @abstractmethod
    def label(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[str, List[str]]:
        raise NotImplementedError()

    def process(
        self,
        elements: List[UiElement],
        loader: Optional[Callable[..., np.ndarray]] = None,
    ):
        images = [e.get_cropped_image(loader) for e in elements]
        labels = self.label(images)
        for e, labels in zip(elements, labels):
            e.info["icon_label"] = labels


class DummyIconLabeller(BaseIconLabeller):
    def label(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> List[str]:
        return []


class ClassifierIconLabeller(BaseIconLabeller):
    model: ModelWrapper
    transform: transforms.Compose
    batched: bool

    def __init__(
        self,
        model_path: str,
        batched: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model = load_model(model_path).to(device)
        self.transform = get_infer_transform(self.model.pretrained)
        self.batched = batched

    def label(
        self, images: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[str, List[str]]:
        # Assume that the images are of shape (h, w, c).
        def inner(image: np.ndarray) -> str:
            transformed = self.transform(_preprocess_image(image))
            _, class_idx = torch.max(self.model(transformed).data, 1)
            return self.model.classes[class_idx.item()]

        def inner_batched(images: List[np.ndarray]) -> List[str]:
            if len(images) == 0:
                return []
            tmp = [self.transform(_preprocess_image(image)) for image in images]
            transformed = torch.cat(tmp, dim=0)
            _, class_indices = torch.max(self.model(transformed).data, -1)
            return [self.model.classes[class_idx.item()] for class_idx in class_indices]

        if isinstance(images, list):
            if self.batched:
                return inner_batched(images)
            return [inner(image) for image in images]
        return inner(images)


class CaptionIconLabeller(BaseIconLabeller):
    model: LabelDroid
    transform: transforms.Compose
    batched: bool

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        batched: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = LabelDroid(
            LabelDroidArgs(model_path=model_path, vocab_path=vocab_path)
        ).to(device)
        self.transform = labeldroid_utils.get_infer_transform()
        self.batched = batched

    def label(
        self, images: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[str, List[str]]:
        def get_sentence(sentence_id: List[int]) -> str:
            tmp = []
            for word_id in sentence_id:
                if self.model.args.vocab is None:
                    raise ValueError("'vocab' has not been set.")
                word = self.model.args.vocab.idx2word[str(word_id)]
                if word == "<end>":
                    break
                tmp.append(word)
            return " ".join(tmp[1:])

        def inner(image: np.ndarray) -> str:
            transformed = self.transform(_preprocess_image(image))
            sentence_ids = self.model(transformed).tolist()
            return get_sentence(sentence_ids[0])

        def inner_batched(images: List[np.ndarray]) -> List[str]:
            if len(images) == 0:
                return []
            tmp = [
                torch.tensor(self.transform(_preprocess_image(image)))
                for image in images
            ]
            transformed = torch.cat(tmp, dim=0)
            sentence_ids = self.model(transformed).tolist()
            return [get_sentence(sentence_id) for sentence_id in sentence_ids]

        if isinstance(images, list):
            if self.batched:
                return inner_batched(images)
            return [inner(image) for image in images]
        return inner(images)


def _preprocess_image(image: np.ndarray) -> torch.Tensor:
    tensor_image = torch.tensor(np.expand_dims(np.transpose(image, (2, 0, 1)), 0))
    if isinstance(tensor_image, torch.ByteTensor):
        return tensor_image.to(torch.float32).div(255)
    return tensor_image.to(torch.float32)
