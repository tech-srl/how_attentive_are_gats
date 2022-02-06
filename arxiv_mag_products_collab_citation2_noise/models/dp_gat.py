from typing import Union, Tuple, Optional
import math

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.inits import glorot


class DotProductGATConv(GATConv):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(DotProductGATConv, self).__init__(in_channels=(in_channels, in_channels), out_channels=out_channels,
                                                heads=heads, concat=concat,
                                                negative_slope=negative_slope, dropout=dropout,
                                                add_self_loops=add_self_loops, bias=bias, **kwargs)
        self.v_layer = nn.Linear(in_channels, heads * out_channels, False)
        glorot(self.v_layer.weight)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'v_layer'):
            glorot(self.v_layer.weight)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_v_l = x_v_r = self.v_layer(x).view(-1, H, C)
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = self.lin_r(x).view(-1, H, C)
        else:
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l, x_r = x[0], x[1]
            x_v_l = self.v_layer(x_l).view(-1, H, C)
            x_v_r = self.v_layer(x_r).view(-1, H, C)
            x_l = self.lin_l(x_l).view(-1, H, C)
            x_r = self.lin_r(x_r).view(-1, H, C)

        alpha_l = x_l
        alpha_r = x_r

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] if size is not None else num_nodes
                num_nodes = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_v_l, x_v_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # alpha_j, alpha_i: [edges, H, C]
        alpha = (alpha_i * alpha_j).sum(-1) / math.sqrt(self.out_channels)  # (edges, H)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
