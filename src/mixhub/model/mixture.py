import sys
import torch
from torch import nn
import torch.nn.functional as F

from mixhub.data.utils import UNK_TOKEN
from mixhub.model.utils import compute_key_padding_mask, ACTIVATION_MAP
from mixhub.model.tensor_types import (
    Tensor,
    MixTensor,
    MaskTensor,
    EmbTensor,
    ManyEmbTensor,
    ManyMixTensor,
    PredictionTensor,
)

from typing import Union, Optional
import torch_geometric as pyg
from torch_geometric.data import Batch
from torch_geometric.nn.conv import FiLMConv
from mixhub.model.predictor import PhysicsPredictiveHead, ScaledCosineRegressor


class MLP(nn.Module):
    """Basic MLP with dropout and GELU activation."""

    def __init__(
        self,
        hidden_dim: int,
        add_linear_last: bool,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(nn.LazyLinear(hidden_dim))
            self.layers.append(nn.GELU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(p=dropout_rate))
        if add_linear_last:
            self.layers.append(nn.LazyLinear(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        output = self.layers(x)
        return output


class FiLMLayer(nn.Module):
    def __init__(
        self,
        output_dim: int,
        act: str,
    ):
        super().__init__()

        self.gamma = nn.LazyLinear(output_dim)
        self.act = ACTIVATION_MAP[act]()
        self.beta = nn.LazyLinear(output_dim)

    def forward(self, x, condition):
        gamma = self.act(self.gamma(condition))
        beta = self.act(self.beta(condition))

        return gamma * x + beta


class AddNorm(nn.Module):
    """Residual connection with layer normalization and dropout."""

    def __init__(self, embed_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.norm(x1 + self.dropout(x2))


class MolecularAttention(nn.Module):
    """Molecule-wise PE attention for a mixture."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        add_mlp: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.self_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.addnorm1 = AddNorm(embed_dim, dropout_rate)
        self.add_mlp = add_mlp
        if self.add_mlp:
            self.ffn = MLP(
                embed_dim, num_layers=1, add_linear_last=True, dropout_rate=0.0
            )
            self.addnorm2 = AddNorm(embed_dim, dropout_rate)

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> MixTensor:
        attn_x, attn_weights = self.self_attn_layer(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )
        out = self.addnorm1(x, attn_x)
        if self.add_mlp:
            out = self.addnorm2(out, self.ffn(out))
        return out, attn_weights


class SelfAttentionBlock(nn.Module):
    """Stack of layers to processes mixtures of molecules with a final aggregation operation."""

    def __init__(
        self,
        mol_aggregation: nn.Module,
        num_layers: int = 1,
        **mol_attn_args,
    ):
        super().__init__()

        layers = (
            [MolecularAttention(**mol_attn_args) for _ in range(num_layers)]
            if num_layers > 0
            else [nn.Identity()]
        )
        self.mol_attn_layers = nn.ModuleList(layers)
        self.mol_aggregation = mol_aggregation
        self.ffn = MLP(
            hidden_dim=mol_attn_args["embed_dim"],
            dropout_rate=mol_attn_args["dropout_rate"],
            add_linear_last=False,
        )

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> EmbTensor:
        for layer in self.mol_attn_layers:
            if isinstance(layer, nn.Identity):
                x = layer(x)
            else:
                x, _ = layer(x, key_padding_mask)

        global_emb = self.mol_aggregation(x, key_padding_mask)
        global_emb = self.ffn(global_emb)
        return global_emb


class DeepSet(nn.Module):
    """Stack of layers to processes mixtures of molecules with a final aggregation operation."""

    def __init__(
        self,
        embed_dim: int,
        mol_aggregation: nn.Module,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.mol_aggregation = mol_aggregation
        self.ffn1 = MLP(
            hidden_dim=embed_dim,
            add_linear_last=False,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
        self.ffn2 = MLP(
            hidden_dim=embed_dim,
            add_linear_last=True,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

    def forward(
        self, x: MixTensor, key_padding_mask: MaskTensor
    ) -> EmbTensor:

        emb = self.ffn1(x)

        global_emb = self.mol_aggregation(emb, key_padding_mask)
        global_emb = self.ffn2(global_emb)

        return global_emb


class MixtureModel(nn.Module):
    def __init__(
        self,
        mol_encoder: nn.Module,
        projection_layer: nn.Module,
        mix_encoder: nn.Module,
        regressor: nn.Module,
        fraction_aggregation_type: str,
        context_aggregation_type: str,
        fraction_film_activation: Optional[str] = None,
        context_film_activation: Optional[str] = None,
        fraction_film_output_dim: Optional[int] = None,
        context_film_output_dim: Optional[int] = None,
    ):
        super(MixtureModel, self).__init__()
        self.mol_encoder = mol_encoder
        self.projection_layer = projection_layer
        self.mix_encoder = mix_encoder
        self.regressor = regressor
        self.unk_token = UNK_TOKEN
        self.fraction_aggregation_type = fraction_aggregation_type
        self.context_aggregation_type = context_aggregation_type

        if self.fraction_aggregation_type == "film":
            self.fraction_film = FiLMLayer(
                output_dim=fraction_film_output_dim,
                act=fraction_film_activation,
            )
        if self.context_aggregation_type == "film":
            self.context_film = FiLMLayer(
                output_dim=context_film_output_dim,
                act=context_film_activation,
            )

    def add_fraction_information(
        self,
        mol_emb: torch.Tensor,
        x_fractions: torch.Tensor,
        fraction_aggregation_type: str,
    ) -> torch.Tensor:

        # Do not add dummy context
        if torch.all(x_fractions == self.unk_token):
            mol_emb = mol_emb
        else:
            if fraction_aggregation_type == "concat":
                mol_emb = torch.concat((mol_emb, x_fractions), dim=-1)
            elif fraction_aggregation_type == "multiply":
                x_fractions = torch.where(x_fractions == self.unk_token, torch.tensor(1.0), x_fractions)
                mol_emb = mol_emb * x_fractions
            elif fraction_aggregation_type == "film":
                padding_mask = compute_key_padding_mask(mol_emb, self.unk_token)
                mol_emb = self.fraction_film(x=mol_emb, condition=x_fractions)
                pad_fill = torch.full_like(mol_emb, self.unk_token)
                mol_emb = torch.where(~padding_mask.unsqueeze(-1) , mol_emb, pad_fill)

        return mol_emb
    
    def add_context_information(
        self,
        mix_emb: torch.Tensor,
        x_context: torch.Tensor,
        context_aggregation_type: str,
    ) -> torch.Tensor:
        
        # Do not add dummy context
        if torch.all(x_context == self.unk_token):
            mix_emb = mix_emb
        else:
            if context_aggregation_type == "concat":
                mix_emb = torch.concat((mix_emb, x_context), dim=-1)
            elif context_aggregation_type == "film":
                mix_emb = self.context_film(x=mix_emb, condition=x_context)

        return mix_emb

    def embed(
        self,
        x: torch.Tensor | pyg.data.Batch,
        x_ids: torch.Tensor,
        x_fractions: torch.Tensor,
        x_context: torch.Tensor,
    ):

        mix_emb_all = []

        for i, x_ids_i in enumerate(torch.unbind(x_ids, dim=-1)):

            if isinstance(x, pyg.data.Batch):
                mol_emb = self.mol_encoder(x, x_ids_i)
            else:
                # TODO: Double check the padding is correct
                padding_mask = compute_key_padding_mask(x[:, :, :, i], self.unk_token)
                mol_emb = self.mol_encoder(x[:, :, :, i])
                pad_fill = torch.full_like(mol_emb, self.unk_token)
                mol_emb = torch.where(~padding_mask.unsqueeze(-1) , mol_emb, pad_fill) 
            
            
            mol_emb = self.add_fraction_information(
                mol_emb=mol_emb,
                x_fractions=x_fractions[:, :, i].unsqueeze(-1),
                fraction_aggregation_type=self.fraction_aggregation_type,
            )

            key_padding_mask = compute_key_padding_mask(mol_emb, self.unk_token)

            mol_emb = self.projection_layer(mol_emb)
            # pad_fill = torch.full_like(mol_emb, self.unk_token)
            # mol_emb = torch.where(~key_padding_mask.unsqueeze(-1) , mol_emb, pad_fill) 

            mix_emb = self.mix_encoder(mol_emb, key_padding_mask)

            if x_context.dim() == 1:
                x_context = x_context.unsqueeze(-1)

            mix_emb = self.add_context_information(
                mix_emb=mix_emb,
                x_context=x_context,
                context_aggregation_type=self.context_aggregation_type,
            )

            mix_emb_all.append(mix_emb)
        
        final_mix_emb = torch.stack(mix_emb_all, dim=-1)

        return final_mix_emb

    def forward(
        self,
        x: torch.Tensor | pyg.data.Batch,
        x_ids: torch.Tensor,
        x_fractions: torch.Tensor,
        x_context: torch.Tensor,
    ) -> PredictionTensor:

        mix_emb = self.embed(
            x=x,
            x_ids=x_ids,
            x_fractions=x_fractions,
            x_context=x_context,
        )

        if isinstance(self.regressor, ScaledCosineRegressor):
            pred = self.regressor(mix_emb)
        else:
            mix_emb = mix_emb.view(mix_emb.size(0), -1)
            if isinstance(self.regressor, PhysicsPredictiveHead):
                pred = self.regressor(mix_emb, x_context)
            else:
                pred = self.regressor(mix_emb)

        return pred