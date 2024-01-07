from abc import abstractmethod
from typing import Callable, List, Optional, Sequence

import torch

from .labeldroid import utils as labeldroid_utils
from .labeldroid.args import LabelDroidArgs
from .labeldroid.models.combined_model import LabelDroid
from ..base import Array, UiElement, UiDetectionModule, array_to_tensor
from ..torchutils import BatchLoader, ModelWrapper, load_model


class BaseIconLabeler(UiDetectionModule):
    """
    A base class for icon labeling modules.

    A class that implements an icon labeling module to be used in the UI detection
    pipeline must inherit this class.
    """

    @abstractmethod
    def label(self, images: Sequence[Array]) -> List[Optional[str]]:
        """Generates labels for icon images

        Parameters
        ----------
        images : Sequence[Array]
            List of icon images.

        Returns
        -------
        List[Optional[str]]
            List of icon labels, each of which is None if not applicable.
        """
        raise NotImplementedError()

    def __call__(
        self,
        elements: List[UiElement],
        loader: Optional[Callable[..., Array]] = None,
    ):
        """Assigns icon labels to the input UI elements

        This function modifies the input UI elements. The icon label of each UI element,
        `element`, is stored with an "icon_label" key in `element.info`.

        Parameters
        ----------
        elements : List[UiElement]
            List of input UI elements.
        loader : Optional[Callable[..., Array]]
            Image loader, used to load the screenshot images of the UI elements.
        """
        self.process(elements, loader)

    def process(
        self,
        elements: List[UiElement],
        loader: Optional[Callable[..., Array]] = None,
    ):
        """Assigns icon labels to the input UI elements

        This function modifies the input UI elements. The icon label of each UI element,
        `element`, is stored with an "icon_label" key in `element.info`.

        Parameters
        ----------
        elements : List[UiElement]
            List of input UI elements.
        loader : Optional[Callable[..., Array]]
            Image loader, used to load the screenshot images of the UI elements.
        """
        images = [e.get_cropped_image(loader) for e in elements]
        labels = self.label(images)
        for e, label in zip(elements, labels):
            if label is None:
                continue
            e.info["icon_label"] = label

    @staticmethod
    def preprocess_image(image: Array) -> torch.Tensor:
        """Preprocesses an image for model's inference

        Parameters
        ----------
        image : Array
            An image to be preprocessed. Must conform to the Array format.

        Returns
        -------
        Tensor
            The resulting PyTorch Tensor.
        """
        tensor_image = array_to_tensor(image)
        return tensor_image


class DummyIconLabeler(BaseIconLabeler):
    """
    A dummy icon labeling module.

    It returns invalid labels for all input images.
    """

    def label(
        self,
        images: Sequence[Array],
    ) -> List[Optional[str]]:
        return [None] * len(images)


class ClassifierIconLabeler(BaseIconLabeler):
    """
    An icon labeling module based on an image classification model.

    Attributes
    ----------
    model : ModelWrapper
        Classification-based icon-labeling model.
    transform : Callable
        Function that transforms the input images to be processed by `model`.
    batch_size : int
        Size of the inference batch.
    """

    model: ModelWrapper
    transform: Callable
    batch_size: int

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to a PyTorch model.
        batch_size : int
            Size of the inference batch.
        device : device
            Device on which the model will run.
        """
        super().__init__()
        self.model = load_model(model_path).to(device)
        self.transform = self.model.transform
        self.batch_size = batch_size

    def label(self, images: Sequence[Array]) -> List[Optional[str]]:
        # Assume that the images are of shape (h, w, c).
        def inner(image: Array) -> str:
            transformed = self.transform(
                self.__class__.preprocess_image(image).unsqueeze(0)
            )
            out = self.model(transformed)
            return self.model.get_preds(out)[0]

        def inner_batched(images: Sequence[Array]) -> List[Optional[str]]:
            if len(images) == 0:
                return []
            transformed = [
                self.transform(self.__class__.preprocess_image(image))
                for image in images
            ]
            result: List[Optional[str]] = []
            loader = BatchLoader(self.batch_size, transformed)
            for batch in loader:
                out = self.model(batch)
                result.extend(self.model.get_preds(out))
            return result

        if self.batch_size == 1:
            return [inner(image) for image in images]
        return inner_batched(images)


class CaptionIconLabeler(BaseIconLabeler):
    """
    An icon labeling module based on an image captioning model.

    This implementation is based on [LabelDroid](https://github.com/chenjshnn/LabelDroid).

    Attributes
    ----------
    model : LabelDroid
        Instance of LabelDroid
    transform : Callable
        Function that transforms the input images to be processed by `model`.
    batch_size : int
        Size of the inference batch.
    """

    model: LabelDroid
    transform: Callable
    batch_size: int

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        model_path : Optional[str]
            Path to a pretrained LabelDroid model.
        vocab_path : Optional[str]
            Path to the vocab of the pretrained model.
        batch_size : int
            Size of the inference batch.
        device : device
            Device on which the model will run.
        """
        self.model = LabelDroid(
            LabelDroidArgs(model_path=model_path, vocab_path=vocab_path)
        ).to(device)
        self.transform = labeldroid_utils.get_infer_transform()
        self.batch_size = batch_size

    def label(self, images: Sequence[Array]) -> List[Optional[str]]:
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

        def inner(image: Array) -> str:
            transformed = self.transform(
                self.__class__.preprocess_image(image).unsqueeze(0)
            )
            sentence_ids = self.model(transformed).tolist()
            return get_sentence(sentence_ids[0])

        def inner_batched(images: Sequence[Array]) -> List[Optional[str]]:
            if len(images) == 0:
                return []
            transformed = [
                self.transform(self.__class__.preprocess_image(image))
                for image in images
            ]
            result: List[Optional[str]] = []
            loader = BatchLoader(self.batch_size, transformed)
            for batch in loader:
                sentence_ids = self.model(batch).tolist()
                result.extend(get_sentence(sentence_id) for sentence_id in sentence_ids)
            return result

        if self.batch_size == 1:
            return [inner(image) for image in images]
        return inner_batched(images)
