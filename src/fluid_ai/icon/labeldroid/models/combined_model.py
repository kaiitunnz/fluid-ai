from typing import Union

import torch
import torch.nn as nn

from ..args import LabelDroidArgs
from . import convcap, lstm, setup, transformer
from .image_models import ResNetFeats


class LabelDroid(nn.Module):
    encoder: ResNetFeats
    decoder: transformer.Transformer
    args: LabelDroidArgs

    def __init__(self, args: LabelDroidArgs):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super().__init__()
        self.encoder = ResNetFeats(**args.get_resnetfeats_args())
        decoder = setup(**args.get_decoder_args())
        if not isinstance(decoder, transformer.Transformer):
            raise NotImplementedError(
                f"'{decoder.__class__.__name__}' is not implemented."
            )
        self.decoder = decoder
        self.args = args

        if args.model_path is not None:
            state_dict = torch.load(args.model_path)
            self.encoder.load_state_dict(state_dict["encoder_state_dict"])
            self.decoder.load_state_dict(state_dict["decoder_state_dict"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(next(self.encoder.parameters()).device)
        features = self.encoder(images)
        sentence_ids = self.decoder.evaluate(features, self.args.max_tokens)
        return sentence_ids
