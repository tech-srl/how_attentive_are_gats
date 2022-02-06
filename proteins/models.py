import math

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.conv.gatconv import GATConv
from gat2 import GAT2Conv
from dpgat import DPGATConv


class GAT(nn.Module):
    def __init__(
        self,
        node_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        activation,
        dropout,
        input_drop,
        attn_drop,
        type
    ):
        super().__init__()

        if type == 'GAT':
            base_layer = GATConv
        elif type == 'GAT2':
            base_layer = GAT2Conv
        elif type == 'DPGAT':
            base_layer = DPGATConv

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_hidden)

        for i in range(n_layers):
            in_hidden = n_hidden
            out_hidden = n_hidden
            # bias = i == n_layers - 1
            if type == 'GAT':
            	layer = base_layer(
                    in_hidden,
                    out_hidden // n_heads,
                    num_heads=n_heads,
                    attn_drop=attn_drop
                )
            else:
            	layer = base_layer(
                    in_hidden,
                    out_hidden // n_heads,
                    num_heads=n_heads,
                    attn_drop=attn_drop,
                    bias=False,
                 	share_weights=True,
                )           	
            self.convs.append(layer)
            self.norms.append(nn.BatchNorm1d(out_hidden))

        self.pred_linear = nn.Linear(out_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]
        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            h = self.convs[i](subgraphs[i], h).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        h = self.pred_linear(h)

        return h

