import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric


class GraphModel(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, dim0, h_dim, out_dim,
                 unroll, layer_norm, use_activation, use_residual, num_heads, dropout):
        super(GraphModel, self).__init__()
        self.gnn_type = gnn_type
        self.unroll = unroll
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_layers = num_layers
        self.layer0_keys = nn.Embedding(num_embeddings=dim0, embedding_dim=h_dim)
        self.layer0_values = nn.Embedding(num_embeddings=dim0, embedding_dim=h_dim)

        self.layer0_ff = nn.Sequential(nn.ReLU())

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        if unroll:
            self.layers.append(gnn_type.get_layer(
                in_dim=h_dim,
                out_dim=h_dim, num_heads=num_heads))
        else:
            for i in range(num_layers):
                self.layers.append(gnn_type.get_layer(
                    in_dim=h_dim,
                    out_dim=h_dim, num_heads=num_heads))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim))

        self.out_dim = out_dim
        self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim, bias=False)

    def forward(self, data, return_attention_weights=None):
        x, edge_index, batch, target_mask = data.x, data.edge_index, data.batch, data.target_mask

        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed
        x = self.layer0_ff(x)

        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]

            new_x = layer(x, edge_index, return_attention_weights=return_attention_weights)
            if return_attention_weights is True:
                new_x, attention_weights = new_x
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = F.dropout(x)

        target_nodes = x[target_mask]
        logits = self.out_layer(target_nodes)
        if return_attention_weights is True:
            return logits, attention_weights
        else:
            targets_batch = batch[target_mask]
            return logits, targets_batch

    def attention_per_edge(self, example):
        logits, (edge_index, alpha) = self.forward(example, return_attention_weights=True)
        _, pred = logits.max(dim=1)

        return pred, alpha.cpu().detach().numpy(), edge_index.cpu().detach().numpy()
