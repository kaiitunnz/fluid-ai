from typing import Union

from . import *
from . import convcap, lstm, transformer


def setup(
    attention: bool = True,
    caption_model: str = "transformer",
    drop_prob_lm: float = 0.1,
    embed_size: int = 512,
    ff_size: int = 2048,
    hidden_size: int = 512,
    img_features_size: int = 2048,
    max_tokens: int = 15,
    numwords: int = 0,
    num_layers: int = 3,
    use_bn: bool = True,
    vocab_len: int = 0,
) -> Union[lstm.DecoderRNN, convcap.convcap, transformer.Transformer]:
    model: Union[lstm.DecoderRNN, convcap.convcap, transformer.Transformer]
    # lstm
    if caption_model == "lstm":
        model = lstm.DecoderRNN(
            embed_size,
            hidden_size,
            vocab_len,
            num_layers,
            max_tokens,
        )
    # convolutional caption
    elif caption_model == "convcap":
        model = convcap.convcap(
            numwords, embed_size, num_layers, is_attention=attention
        )
    # Transformer
    elif caption_model == "transformer":
        model = transformer.Transformer(
            img_features_size,
            embed_size,
            use_bn,
            drop_prob_lm,
            vocab_len,
            num_layers,
            ff_size,
        )
    else:
        raise Exception("Caption model not supported: {}".format(caption_model))

    return model
