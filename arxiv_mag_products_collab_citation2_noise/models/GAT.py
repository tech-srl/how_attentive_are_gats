from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.dp_gat import DotProductGATConv
from models.gat2 import GAT2Conv
# from models.dp_gat_new import DotProductGATConv
from tqdm import tqdm


class GAT(torch.nn.Module):
    def __init__(self, base_layer, in_channels, hidden_channels, out_channels, num_layers, num_heads,
                 dropout, device, saint, use_layer_norm, use_residual, use_resdiual_linear):
        super(GAT, self).__init__()

        self.layers = torch.nn.ModuleList()
        kwargs = {
            'bias':False
        }
        if base_layer is GAT2Conv:
            kwargs['share_weights'] = True
        self.layers.append(base_layer(in_channels, hidden_channels // num_heads, num_heads, **kwargs))
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_resdiual_linear = use_resdiual_linear
        self.layer_norms = torch.nn.ModuleList()
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        self.residuals = torch.nn.ModuleList()
        if use_resdiual_linear and use_residual:
            self.residuals.append(nn.Linear(in_channels, hidden_channels))
        self.num_layers = num_layers
        for _ in range(num_layers - 2):
            self.layers.append(
                base_layer(hidden_channels, hidden_channels // num_heads, num_heads, **kwargs))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
            if use_resdiual_linear and use_residual:
                self.residuals.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(base_layer(hidden_channels, out_channels, 1, **kwargs ))
        if use_resdiual_linear and use_residual:
            self.residuals.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout
        self.device = device
        self.saint = saint
        self.non_linearity = F.relu
        print(f"learnable_params: {sum(p.numel() for p in list(self.parameters()) if p.requires_grad)}")

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.layer_norms:
            layer.reset_parameters()
        for layer in self.residuals:
            layer.reset_parameters()

    def forward_neighbor_sampler(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            new_x = self.layers[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                new_x = self.non_linearity(new_x)
            if 0 < i < self.num_layers - 1 and self.use_residual:
                x = new_x + x_target
            else:
                x = new_x
            if i < self.num_layers - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def exp_forward_neighbor_sampler(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            new_x = self.layers[i]((x, x_target), edge_index)
            if self.use_residual:
                if self.use_resdiual_linear:
                    x = new_x + self.residuals[i](x_target)
                else:
                    x = new_x + x_target
            else:
                x = new_x

            if i < self.num_layers - 1:
                x = self.non_linearity(x)
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_saint(self, x, adj_t):
        for i, layer in enumerate(self.layers[:-1]):
            new_x = layer(x, adj_t)
            new_x = self.non_linearity(new_x)
            # residual
            if i > 0 and self.use_residual:
                if self.use_resdiual_linear:
                    x = new_x + self.residuals[i](x)
                else:
                    x = new_x + x
                x = new_x + x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_t)
        return x

    def forward(self, x, adjs):
        if self.saint:
            return self.forward_saint(x, adjs)
        else:
            return self.forward_neighbor_sampler(x, adjs)

    def inference(self, x, subgraph_loader):
        pbar = tqdm(total=x.size(0) * len(self.layers), leave=False, desc="Layer", disable=False)
        pbar.set_description('Evaluating')
        for i, layer in enumerate(self.layers[:-1]):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x_source = x[n_id].to(self.device)
                x_target = x_source[:size[1]]  # Target nodes are always placed first.
                new_x = layer((x_source, x_target), edge_index)
                new_x = self.non_linearity(new_x)
                # residual
                if i > 0 and self.use_residual:
                    x_target = new_x + x_target
                else:
                    x_target = new_x
                if self.use_layer_norm:
                    x_target = self.layer_norms[i](x_target)
                # x_target = F.dropout(x_target, p=self.dropout, training=self.training)
                xs.append(x_target.cpu())
                pbar.update(batch_size)
            x = torch.cat(xs, dim=0)
        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj.to(self.device)
            x_source = x[n_id].to(self.device)
            x_target = x_source[:size[1]]  # Target nodes are always placed first.
            new_x = self.layers[-1]((x_source, x_target), edge_index)
            xs.append(new_x.cpu())
            pbar.update(batch_size)
        x = torch.cat(xs, dim=0)
        pbar.close()
        return x

    def exp_inference(self, x, subgraph_loader):
        pbar = tqdm(total=x.size(0) * len(self.layers), leave=False, desc="Layer", disable=False)
        pbar.set_description('Evaluating')
        for i, layer in enumerate(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(self.device)
                x_source = x[n_id].to(self.device)
                x_target = x_source[:size[1]]  # Target nodes are always placed first.
                new_x = layer((x_source, x_target), edge_index)
                if self.use_residual:
                    if self.use_resdiual_linear:
                        x_target = new_x + self.residuals[i](x_target)
                    else:
                        x_target = new_x + x_target
                else:
                    x_target = new_x
                if i < self.num_layers - 1:
                    x_target = self.non_linearity(x_target)
                    if self.use_layer_norm:
                        x_target = self.layer_norms[i](x_target)

                xs.append(x_target.cpu())
                pbar.update(batch_size)
            x = torch.cat(xs, dim=0)
        pbar.close()
        return x


class GAT_TYPE(Enum):
    GAT = auto()
    DPGAT = auto()
    GAT2 = auto()

    @staticmethod
    def from_string(s):
        try:
            return GAT_TYPE[s]
        except KeyError:
            raise ValueError()
    
    def __str__(self):
        if self is GAT_TYPE.GAT:
            return "GAT"
        elif self is GAT_TYPE.DPGAT:
            return "DPGAT"
        elif self is GAT_TYPE.GAT2:
            return "GAT2"
        return "NA"

    def get_model(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device, saint, use_layer_norm, use_residual, use_resdiual_linear):
        if self is GAT_TYPE.GAT:
            return GAT(GATConv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device, saint, use_layer_norm, use_residual, use_resdiual_linear)
        elif self is GAT_TYPE.DPGAT:
            return GAT(DotProductGATConv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device, saint, use_layer_norm, use_residual, use_resdiual_linear)
        elif self is GAT_TYPE.GAT2:
            return GAT(GAT2Conv, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, device, saint, use_layer_norm, use_residual, use_resdiual_linear)

    def get_base_layer(self):
        if self is GAT_TYPE.GAT:
            return GATConv
        elif self is GAT_TYPE.DPGAT:
            return DotProductGATConv
        elif self is GAT_TYPE.GAT2:
            return GAT2Conv
