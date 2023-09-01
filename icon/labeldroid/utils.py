import json
import os

from torchvision import transforms

from .args import LabelDroidArgs
from .models.combined_model import LabelDroid
from .vocab import Vocabulary

IMAGE_SIZE = 224


def build_vocab_from_json(idx2word_path: str, word2idx_path: str) -> Vocabulary:
    vocab = Vocabulary()
    with open(os.path.join(idx2word_path), "r") as f:
        vocab.idx2word = json.loads(f.read())
    with open(os.path.join(word2idx_path), "r") as f:
        vocab.word2idx = json.loads(f.read())
    vocab.idx = max(vocab.idx2word.keys())
    return vocab


def load_default_model(model_path: str, vocab_path: str) -> LabelDroid:
    args = LabelDroidArgs(model_path, vocab_path)
    return LabelDroid(args)


def get_infer_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
