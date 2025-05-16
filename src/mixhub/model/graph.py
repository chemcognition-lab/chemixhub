from typing import Union, Optional, List

import json
import numpy as np

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import MetaLayer, Linear, GAT
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.data import Batch

from mixhub.data.utils import UNK_TOKEN

# and (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html#torch_geometric.nn.models.MetaLayer)


class EdgeFiLMModel(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        hidden_dim: Optional[int] = 50,
        num_layers: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # FiLM
        self.gamma = get_mlp(hidden_dim, edge_dim, num_layers, dropout=dropout)
        self.gamma_act = nn.Sigmoid()  # sigmoidal gating Dauphin et al.
        self.beta = get_mlp(hidden_dim, edge_dim, num_layers, dropout=dropout)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        cond = torch.cat([src, dst, u[batch]], 1)
        gamma = self.gamma_act(self.gamma(cond))
        beta = self.beta(cond)

        return gamma * edge_attr + beta


class NodeAttnModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: Optional[int] = 50,
        num_heads: Optional[int] = 5,
        dropout: Optional[int] = 0.0,
        num_layers: Optional[int] = 1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers

        # self attention layer
        self.self_attn = GAT(
            node_dim,
            node_dim,
            num_layers=num_layers,
            dropout=dropout,
            v2=True,
            heads=num_heads,
        )
        self.output_mlp = get_mlp(hidden_dim, node_dim, num_layers=2)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        attn = self.self_attn(x, edge_index, edge_attr)
        out = self.norm1(x + self.dropout_layer(attn))
        out = self.norm2(out + self.dropout_layer(self.output_mlp(out)))
        return out


class GlobalPNAModel(nn.Module):
    def __init__(
        self,
        global_dim: int,
        hidden_dim: Optional[int] = 50,
        num_layers: Optional[int] = 2,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.pool = MultiAggregation(["mean", "std", "max", "min"])
        self.global_mlp = get_mlp(hidden_dim, global_dim, num_layers, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        aggr = self.pool(x, batch)
        out = torch.cat([u, aggr], dim=1)
        return self.global_mlp(out)


##### Helper functions #####


def get_mlp(
    hidden_dim: int, output_dim: int, num_layers: int, dropout: Optional[float] = 0.0
):
    """
    Helper function to produce MLP with specified hidden dimension and layers
    """
    assert num_layers > 0, "Enter an integer larger than 0."

    layers = nn.ModuleList()
    for _ in range(num_layers - 1):
        layers.append(Linear(-1, hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.SELU())
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(Linear(-1, output_dim))
    return nn.Sequential(*layers)


def get_graphnet_layer(
    node_dim: int,
    edge_dim: int,
    global_dim: int,
    hidden_dim: Optional[int] = 50,
    dropout: Optional[float] = 0.0,
):
    """
    Helper function to produce GraphNets layer.
    """
    node_net = NodeAttnModel(node_dim, hidden_dim=hidden_dim, dropout=dropout)
    edge_net = EdgeFiLMModel(edge_dim, hidden_dim=hidden_dim, dropout=dropout)
    global_net = GlobalPNAModel(global_dim, hidden_dim=hidden_dim, dropout=dropout)
    return MetaLayer(edge_net, node_net, global_net)


class GraphNets(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        task_type: str = "multi-component",
        hidden_dim: Optional[int] = 50,
        depth: Optional[int] = 3,
        dropout_rate: Optional[float] = 0.0,
        **kwargs,
    ):
        super(GraphNets, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList(
            [
                get_graphnet_layer(
                    node_dim,
                    edge_dim,
                    global_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout_rate,
                )
                for _ in range(depth)
            ]
        )

        supported_task_types = ["single-component", "multi-component"]
        if task_type in supported_task_types:
            self.task_type = task_type
        else:
            raise ValueError(f"{task_type} not in {supported_task_types}")

    def forward_single_component(self, data: pyg.data.Data):
        x, edge_index, edge_attr, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            data.batch,
        )

        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)

        return u

    # def graphs_to_mixtures(
    #     self,
    #     data: pyg.data.Data,
    #     collate_indices: Union[torch.Tensor, np.ndarray],
    #     device: Optional[Union[torch.device, str]] = "cpu",
    #     unk_token: Optional[int] = -999,
    # ):
    #     device = torch.device(device)
    #     data = data.to(device)
    #     x = self.forward(
    #         data
    #     )  # produces molecule embeddings [num_unique_mols, embed_dim]
    #     padding = torch.full((1, x.shape[-1]), unk_token, device=device)
    #     x = torch.cat([x, padding])  # [num_unique_mols + 1, embed_dim]

    #     out = torch.stack(
    #         [x[collate_indices[:, i]] for i in range(collate_indices.shape[1])], dim=1
    #     )  # [num_samples, max_mols*num_mix, embed_dim]
    #     out = out.reshape(collate_indices.shape + (x.shape[-1],))
    #     out = torch.transpose(out, -2, -1)

    #     return out

    def forward_multi_component(
            self,
            data: List[pyg.data.Data],
            data_indices: torch.Tensor,
        ):

        mol_emb = self.forward_single_component(data)

        output = torch.full(
            (data_indices.shape[0], data_indices.shape[1], mol_emb.shape[-1]),
            fill_value=UNK_TOKEN,
            device=mol_emb.device,
            dtype=mol_emb.dtype
        )

        # Create mask for valid indices
        valid_mask = data_indices != UNK_TOKEN

        # Get valid positions and corresponding embeddings
        valid_indices = data_indices[valid_mask]  # shape: (num_valid,)
        output[valid_mask] = mol_emb[valid_indices]  # broadcasted to (num_valid, emb_size)

        return output

    def forward(
        self,
        data: Union[pyg.data.Data, List[List[pyg.data.Data]]],
        data_indices: torch.Tensor,
    ):

        # TODO: add testing guard for eahc data mode to flow in each forward separately
        if self.task_type == "single-component":
            return self.forward_single_component(data=data)
        elif self.task_type == "multi-component":
            return self.forward_multi_component(data=data, data_indices=data_indices)

    @classmethod
    def from_json(cls, node_dim, edge_dim, json_path: str):
        params = json.load(open(json_path, "r"))
        return cls(node_dim, edge_dim, **params)
