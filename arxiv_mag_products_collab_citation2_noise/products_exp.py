# Reaches around 0.7945 ± 0.0059 test accuracy.

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv
import argparse
from models.GAT import GAT_TYPE
from utils.logger import Logger
import random

parser = argparse.ArgumentParser(description='OGBN - products (GAT/DPGAT)')
parser.add_argument("--type", dest="type", default=GAT_TYPE.DPGAT, type=GAT_TYPE.from_string, choices=list(GAT_TYPE))
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--eval_steps', type=int, default=10)
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--patient', type=int, default=10)
parser.add_argument('--max_loss', type=float, default=2.0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()
print(args)
torch.manual_seed(args.seed)
random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = PygNodePropPredDataset('ogbn-products')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[10] * args.num_layers, batch_size=256,
                               shuffle=True, num_workers=6)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=6)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        base_layer = args.type.get_base_layer()
        self.convs = torch.nn.ModuleList()
        kwargs = {
            'bias':False
        }
        if args.type is GAT_TYPE.GAT2:
            kwargs['share_weights'] = True
        self.convs.append(base_layer(in_channels, hidden_channels // heads,
                                  heads, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(
                base_layer( hidden_channels, hidden_channels // heads, heads, **kwargs))
        self.convs.append(
            base_layer(hidden_channels, out_channels, heads=1, **kwargs))
        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels, hidden_channels))
        self.skips.append(Lin(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=args.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)
                # x = x + x_target

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all



model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes, num_layers=args.num_layers,
            heads=args.num_heads)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0), leave=False, dynamic_ncols=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

logger = Logger(10, args)
run = 0
while run < args.runs:
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_success = True
    low_loss = False
    max_val_score = -float('inf')
    patient = 0
    best_val_acc = best_train_acc = final_test_acc = 0
    logger.set_time(run)
    for epoch in range(1, 1 + args.epochs):
        loss, acc = train(epoch)
        # print(f'Run: {run:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        if loss <= args.max_loss:
            low_loss = True
        if epoch > args.epochs // 2 and loss > args.max_loss and low_loss is False:
            run_success = False
            logger.reset(run)
            print('Learning failed. Rerun...')
            break
        if epoch > args.epochs // 2 and epoch % args.eval_steps == 0:
        # if epoch > 50 and epoch % 10 == 0:
            result= test()
            train_acc, valid_acc, test_acc = result
            logger.add_result(run, result)
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
            if valid_acc >= max_val_score:
                max_val_score = valid_acc
                patient = 0
            else:
                patient += 1
                if patient >= args.patient:
                    print("patient exhausted")
                    if low_loss is False:
                        run_success = False
                        logger.reset(run)
                        print('Learning failed. Rerun...')
                    break
        elif epoch % args.log_steps == 0:
            print(f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f},'
                f'Approx. Train: {acc:.4f}')
    if run_success:
        logger.print_statistics(run)
        run += 1
logger.print_statistics()