import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler,  GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
import argparse
from utils.logger import Logger
from models.GAT import GAT_TYPE
import torch_geometric.transforms as T
from tqdm import tqdm
from utils.edge_noise import add_edge_noise
import random

def train(model, data, loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    if type(loader) is GraphSAINTRandomWalkSampler:
        for data in tqdm(loader, leave=False, desc=f"Epcoh {epoch}", dynamic_ncols=True):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            y = data.y.squeeze(1)
            loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)
    else:
        for batch_size, n_id, adjs in tqdm(loader, leave=False, desc=f"Epcoh {epoch}", dynamic_ncols=True):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            x = data.x[n_id].to(device)
            y = data.y[n_id[:batch_size]].squeeze().to(device)

            optimizer.zero_grad()
            out = model(x, adjs)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
    return total_loss / len(loader)


@torch.no_grad()
def test(model, data, loader, split_idx, evaluator, metric):
    model.eval()

    out = model.inference(data.x, subgraph_loader=loader)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })[metric]
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })[metric]
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })[metric]

    return train_acc, valid_acc, test_acc


def get_objs(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    res = dict()
    res['device'] = device

    if args.dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToUndirected())
        res['dataset'] = dataset
        data = dataset[0]
        res['data'] = data
        res['split_idx'] = dataset.get_idx_split()
        res['train_idx'] = res['split_idx']['train']
        res['model'] = args.type.get_model(data.num_features, args.hidden_channels,
                                    dataset.num_classes, args.num_layers, args.num_heads,
                                    args.dropout, device, args.use_saint, args.use_layer_norm, args.use_residual, args.use_resdiual_linear).to(device)
        res['evaluator'] = Evaluator(name='ogbn-arxiv')
        res['metric'] = 'acc'
    elif args.dataset == 'ogbn-products':
        raise NotImplementedError
    elif args.dataset == 'ogb-proteins':
        raise NotImplementedError
    elif args.dataset == 'ogbn-mag':
        dataset = PygNodePropPredDataset(name='ogbn-mag')
        res['dataset'] = dataset
        rel_data = dataset[0]
        # We are only interested in paper <-> paper relations.
        data = Data(
            x=rel_data.x_dict['paper'],
            edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
            y=rel_data.y_dict['paper'])
        data = T.ToUndirected()(data)
        res['data'] = data
        res['split_idx'] = {k: v['paper'] for k,v in dataset.get_idx_split().items()}
        res['train_idx'] = res['split_idx']['train'].to(device)
        res['model'] = args.type.get_model(data.num_features, args.hidden_channels,
                                    dataset.num_classes, args.num_layers, args.num_heads,
                                    args.dropout, device, args.use_saint, args.use_layer_norm, args.use_residual, args.use_resdiual_linear).to(device)
        res['evaluator'] = Evaluator(name='ogbn-mag')
        res['metric'] = 'acc'
    else:
        raise AttributeError
    return res


def main():
    parser = argparse.ArgumentParser(description='OGBN (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument("--type", dest="type", default=GAT_TYPE.GAT, type=GAT_TYPE.from_string, choices=list(GAT_TYPE))
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-mag',
                                                                              'ogbn-proteins'])
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--use_saint', action='store_true', default=False)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--use_layer_norm', action='store_true', default=True)
    parser.add_argument('--use_residual', action='store_true', default=True)
    parser.add_argument('--use_resdiual_linear', action='store_true', default=False)
    parser.add_argument('--noise_level', type=float, default=0)
    parser.add_argument('--patient', type=int, default=float('inf'))
    parser.add_argument('--max_loss', type=float, default=float('inf'))
    parser.add_argument('--min_epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    objs = get_objs(args)
    dataset = objs['dataset']
    data = objs['data']
    split_idx = objs['split_idx']
    train_idx = objs['train_idx']
    model = objs['model']
    evaluator = objs['evaluator']
    device = objs['device']
    metric = objs['metric']

    if args.noise_level > 0:
        data.edge_index = add_edge_noise(data.edge_index.to(device), data.x.size(0), args.noise_level)

    if args.use_saint:
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{key}_mask'] = mask
        train_loader = GraphSAINTRandomWalkSampler(data,
                                                   batch_size=args.batch_size,
                                                   walk_length=args.walk_length,
                                                   num_steps=args.num_steps,
                                                   sample_coverage=0,
                                                   save_dir=dataset.processed_dir)
    else:
        train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=args.num_layers * [10],
                                       batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=4096, shuffle=False, num_workers=args.num_workers)

    logger = Logger(args.runs, args)
    print(f"learnable_params: {sum(p.numel() for p in list(model.parameters()) if p.requires_grad)}")
    run = 0
    while run < args.runs:
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        run_success = True
        low_loss = False
        max_val_score = -float('inf')
        patient = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_loader, optimizer, device, epoch)
            if loss <= args.max_loss:
                low_loss = True
            if epoch > args.epochs // 2 and loss > args.max_loss and low_loss is False:
                run_success = False
                logger.reset(run)
                print('Learning failed. Rerun...')
                break
            # if epoch > 50 and epoch % 10 == 0:
            if epoch % args.eval_steps == 0:
                result = test(model, data, subgraph_loader, split_idx, evaluator, metric)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                if args.log_steps:
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
                        if low_loss is False or epoch < args.min_epoch:
                            run_success = False
                            logger.reset(run)
                            print('Learning failed. Rerun...')
                        break
            elif epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}')

        if run_success:
            logger.print_statistics(run)
            run += 1
    logger.print_statistics()


if __name__ == "__main__":
    main()
