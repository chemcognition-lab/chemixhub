import enum
import functools
import sys
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from mixhub.model.tensor_types import *
from mixhub.model.maths_ops import masked_max, masked_mean, masked_min, masked_variance, _LARGENEG_NUM

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class MeanAggregation(nn.Module):
    """Simple mean aggregation with masking."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> EmbTensor:
        global_emb = masked_mean(x, ~key_padding_mask)
        global_emb = global_emb.squeeze(1)
        return global_emb


class MaxAggregation(nn.Module):
    """Simple max aggregation with masking."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> EmbTensor:
        global_emb = masked_max(x, ~key_padding_mask).unsqueeze(1)
        global_emb = global_emb.squeeze(1)
        return global_emb


class Set2SetAggregation(nn.Module):
    """
    modified from: https://github.com/arunppsg/Set2Set/blob/main/set2set/set2set.py

    Set2Set model is used for learning order invariant representation
    of vectors which can later be used with Seq2Seq models or representing
    vertices/edges of a graph to a feed-forward neural network.

    Args:
        in_channels: The output size of the embedding representing the elements of the set
        processing_steps: The number of steps of computation to perform over the elements of the set 
        num_layers: Number of recurrent layers to use in LSTM

    Inputs:
        x: tensor of shape :math:`(L, N, in_channels)` where L is the number of batches,
            N denotes number of elements in the batch and in_channels is the dimension of
            the element in the batch
    Outputs:
        q: tensor of shape :math:`(L, in_channels)` where L is the number of batches,
            in_channels is the embedding size
    """
    def __init__(
        self,
        in_channels: int,
        processing_steps: int,
        num_layers: int = 1,
    ):
        super(Set2SetAggregation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.out_channels,
                            hidden_size=self.in_channels,
                            num_layers=self.num_layers)
        self.projection_layer = nn.LazyLinear(in_channels)

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor,
    ) -> EmbTensor:
        batch_size = x.size()[0]
        n = x.size()[1]
        hidden = (torch.zeros(self.num_layers, batch_size, self.in_channels, device=x.device),
                  torch.zeros(self.num_layers, batch_size, self.in_channels, device=x.device))
        q_star = torch.zeros(1, batch_size, self.out_channels, device=x.device)

        x_masked = torch.einsum("ijk, ij -> ijk", x, ~key_padding_mask)

        for i in range(self.processing_steps):
            # q_star: batch_size * out_channels
            q, hidden = self.lstm(q_star, hidden)
            e = torch.einsum("kij,ibj->kib", q, x_masked)

            e_masked = e.masked_fill(key_padding_mask.unsqueeze(0), _LARGENEG_NUM)

            # e: 1 x batch_size x n
            a = nn.Softmax(dim=2)(e_masked).squeeze(0)

            r = torch.einsum('ij,ijk->ijk', a, x_masked).sum(axis=1)

            # r: 1 x batch_size x n
            q_star = torch.cat([q, r.unsqueeze(0)], dim=-1)
        
        # q_star = self.projection_layer(q_star)

        return q_star.squeeze(0)


class PrincipalNeighborhoodAggregation(nn.Module):
    """PN-style (mean, var, min, max) aggregation."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> EmbTensor:
        mean = masked_mean(x, ~key_padding_mask).unsqueeze(1)
        var = masked_variance(x, ~key_padding_mask).unsqueeze(1)
        minimum = masked_min(x, ~key_padding_mask).unsqueeze(1)
        maximum = masked_max(x, ~key_padding_mask).unsqueeze(1)
        global_emb = torch.cat((mean, var, minimum, maximum), dim=-1)
        global_emb = global_emb.squeeze(1)
        return global_emb


class PNAMixtureSizeScaled(nn.Module):
    """PN-style (mean, var, min, max) aggregation."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> EmbTensor:
        mean = masked_mean(x, ~key_padding_mask).unsqueeze(1)
        var = masked_variance(x, ~key_padding_mask).unsqueeze(1)
        minimum = masked_min(x, ~key_padding_mask).unsqueeze(1)
        maximum = masked_max(x, ~key_padding_mask).unsqueeze(1)
        global_emb = torch.cat((mean, var, minimum, maximum), dim=-1)

        num_components = (~key_padding_mask).sum(dim=1)
        scaling_factor = 1.0 / num_components.float()

        scaled_global_emb = global_emb * scaling_factor.view(global_emb.shape[0], 1, 1) 

        global_emb = torch.cat((global_emb, scaled_global_emb), dim=-1)
        global_emb = global_emb.squeeze(1)

        return global_emb
    

class AttentionAggregation(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.cross_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.mean_agg = MeanAggregation()

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> EmbTensor:
        avg_emb = self.mean_agg(x, key_padding_mask).unsqueeze(1)
        global_emb, _ = self.cross_attn_layer(
            query=avg_emb,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        global_emb = global_emb.squeeze(1)
        return global_emb

