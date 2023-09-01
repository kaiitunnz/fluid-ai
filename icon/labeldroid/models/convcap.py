import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def Conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    dropout: float = 0,
) -> nn.Conv1d:
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Embedding(
    num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]
) -> nn.Embedding:
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features: int, out_features: int, dropout: float = 0.0) -> nn.Linear:
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class AttentionLayer(nn.Module):
    in_projection: nn.Linear
    out_projection: nn.Linear
    bmm: torch.Tensor

    def __init__(self, conv_channels: int, embed_dim: int):
        super().__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        self.out_projection = Linear(embed_dim, conv_channels)
        self.bmm = torch.bmm

    def forward(
        self, x: torch.Tensor, wordemb: torch.Tensor, imgsfeats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x

        x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)

        b, c, f_h, f_w = imgsfeats.size()
        y = imgsfeats.view(b, c, f_h * f_w)

        x = self.bmm(x, y)

        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=0)
        x = x.view(sz)
        attn_scores = x

        y = y.permute(0, 2, 1)

        x = self.bmm(x, y)

        s = y.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        x = (self.out_projection(x) + residual) * math.sqrt(0.5)

        return x, attn_scores


class convcap(nn.Module):
    attention: nn.ModuleList
    classifier_0: nn.Linear
    classifier_1: nn.Linear
    convs: nn.ModuleList
    dropout: float
    emb_0: nn.Embedding
    emb_1: nn.Linear
    imgproj: nn.Linear
    is_attention: bool
    kernel_size: int
    nimgfeats: int
    nfeats: int
    n_layers: int
    pad: int
    resproj: nn.Linear

    def __init__(
        self,
        num_wordclass: int,
        embed_size: int = 4096,
        num_layers: int = 1,
        is_attention: bool = True,
        nfeats: int = 2048,
        dropout: float = 0.1,
    ):
        super(convcap, self).__init__()
        self.nimgfeats = embed_size
        self.is_attention = is_attention
        self.nfeats = nfeats
        self.dropout = dropout

        self.emb_0 = Embedding(num_wordclass, nfeats, padding_idx=0)
        self.emb_1 = Linear(nfeats, nfeats, dropout=dropout)

        self.imgproj = Linear(self.nimgfeats, self.nfeats, dropout=dropout)
        self.resproj = Linear(nfeats * 2, self.nfeats, dropout=dropout)

        n_in = 2 * self.nfeats
        n_out = self.nfeats
        self.n_layers = num_layers
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.kernel_size = 5
        self.pad = self.kernel_size - 1
        for _ in range(self.n_layers):
            self.convs.append(
                Conv1d(n_in, 2 * n_out, self.kernel_size, self.pad, dropout)
            )
            if self.is_attention:
                self.attention.append(AttentionLayer(n_out, nfeats))
            n_in = n_out

        self.classifier_0 = Linear(self.nfeats, (nfeats // 2))
        self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)

    def forward(
        self, imgsfeats: torch.Tensor, imgsfc7: torch.Tensor, wordclass: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_buffer = None
        wordemb = self.emb_0(wordclass)
        wordemb = self.emb_1(wordemb)
        x = wordemb.transpose(2, 1)
        batchsize, wordembdim, maxtokens = x.size()

        y = F.relu(self.imgproj(imgsfc7))
        y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
        x = torch.cat([x, y], 1)

        for i, conv in enumerate(self.convs):
            if i == 0:
                x = x.transpose(2, 1)
                residual = self.resproj(x)
                residual = residual.transpose(2, 1)
                x = x.transpose(2, 1)
            else:
                residual = x

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = conv(x)
            x = x[:, :, : -self.pad]

            x = F.glu(x, dim=1)

            if self.is_attention:
                attn = self.attention[i]
                x = x.transpose(2, 1)
                x, attn_buffer = attn(x, wordemb, imgsfeats)
                x = x.transpose(2, 1)

            x = (x + residual) * math.sqrt(0.5)

        x = x.transpose(2, 1)

        x = self.classifier_0(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier_1(x)

        x = x.transpose(2, 1)

        return x, attn_buffer

    def evaluate(
        self,
        imgsfeats: torch.Tensor,
        imgsfc7: torch.Tensor,
        max_tokens: int,
        start_symbol: int = 1,
    ):
        batch_size = imgsfeats.shape[0]
        sentence_ids = np.zeros((batch_size, max_tokens), dtype="int64")
        sentence_ids[:, 0] = start_symbol

        for j in range(max_tokens - 2):
            wordclass = Variable(torch.from_numpy(sentence_ids)).cuda()

            wordact, _ = self.forward(imgsfeats, imgsfc7, wordclass)
            # batchsize * voc_len * seq_len
            wordact = wordact[:, :, :-1]
            wordact_t = wordact.permute(0, 2, 1).contiguous()

            wordprobs = F.softmax(wordact_t, dim=2).cpu().data.numpy()
            next_words = np.squeeze(np.argmax(wordprobs, axis=2)).astype(
                sentence_ids.dtype
            )
            sentence_ids[:, j + 1] = next_words[:, j + 1]
        return sentence_ids
