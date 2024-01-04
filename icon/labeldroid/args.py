from dataclasses import dataclass
from typing import Any, Dict, Optional

from .vocab import Vocabulary


@dataclass
class LabelDroidArgs:
    """
    Arguments to construct a LabelDroid model.

    Attributes
    ----------
    model_path : Optional[str]
        Path to a saved model.
    vocab_path : Optional[str]
        Path to the vocabulary file.
    """

    model_path: Optional[str] = None
    vocab_path: Optional[str] = None

    # ResNetFeats
    att_size: int = 7
    caption_model: str = "transformer"
    finetune_cnn: bool = False
    img_embed_size: int = 512

    # Decoder
    attention: bool = True
    drop_prob_lm: float = 0.1
    embed_size: int = 512
    ff_size: int = 2048
    hidden_size: int = 512
    img_features_size: int = 2048
    max_tokens: int = 15
    numwords: Optional[int] = None
    num_layers: int = 3
    use_bn: Optional[bool] = None
    vocab: Optional[Vocabulary] = None

    def __post_init__(self):
        if self.vocab_path is not None:
            self.vocab = Vocabulary.load(self.vocab_path)

    def get_resnetfeats_args(self) -> Dict[str, Any]:
        """Gets features/hyperparameters of the ResNet model

        Returns
        -------
        Dict[str, Any]
            Mapping from hyperparameter names to values.
        """
        return {
            "caption_model": self.caption_model,
            "att_size": self.att_size,
            "embed_size": self.img_embed_size,
            "finetune_cnn": self.finetune_cnn,
        }

    def get_decoder_args(self) -> Dict[str, Any]:
        """Gets features/hyperparameters of the decoder

        Returns
        -------
        Dict[str, Any]
            Mapping from hyperparameter names to values.
        """
        if self.vocab is None:
            raise ValueError("`vocab` must be set.")
        return {
            "attention": self.attention,
            "caption_model": self.caption_model,
            "drop_prob_lm": self.drop_prob_lm,
            "embed_size": self.embed_size,
            "ff_size": self.ff_size,
            "hidden_size": self.hidden_size,
            "img_features_size": self.img_features_size,
            "max_tokens": self.max_tokens,
            "numwords": self.numwords,
            "num_layers": self.num_layers,
            "use_bn": self.use_bn,
            "vocab_len": len(self.vocab),
        }
