import argparse
import os
import random
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch import nn

from models import GAT
from utils import BatchSampler, DataLoaderWrapper

from tqdm import tqdm

dataset = "ogbn-proteins"
n_node_feats, n_classes = 0, 112


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset):
    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph, labels, train_idx):
    global n_node_feats

    # The sum of the weights of adjacent edges is used as node features.
    graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))
    n_node_feats = graph.ndata["feat"].shape[-1]

    # Only the labels in the training set are used as features, while others are filled with zeros.
    graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes)
    graph.ndata["train_labels_onehot"][train_idx, labels[train_idx, 0]] = 1

    graph.create_formats_()

    return graph, labels


def gen_model(args):
    n_node_feats_ = n_node_feats

    model = GAT(
        n_node_feats_,
        n_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        type=args.type
    )

    return model


def train(args, model, dataloader, _labels, _train_idx, criterion, optimizer, _evaluator, epoch, device):
    model.train()

    loss_sum, total = 0, 0

    for input_nodes, output_nodes, subgraphs in tqdm(dataloader, leave=False, desc=f"Training epoch {epoch}"):
        subgraphs = [b.to(device) for b in subgraphs]
        new_train_idx = torch.arange(len(output_nodes))

        train_pred_idx = new_train_idx

        pred = model(subgraphs)
        loss = criterion(pred[train_pred_idx], subgraphs[-1].dstdata["labels"][train_pred_idx].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = len(train_pred_idx)
        loss_sum += loss.item() * count
        total += count

        torch.cuda.empty_cache()

    return loss_sum / total


@torch.no_grad()
def evaluate(args, model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator, epoch, device):
    model.eval()

    preds = torch.zeros(labels.shape).to(device)

    # Due to the limitation of memory capacity, we calculate the average of logits 'eval_times' times.
    eval_times = 1

    for _ in range(eval_times):
        for input_nodes, output_nodes, subgraphs in tqdm(dataloader, leave=False, desc=f"Evaluating epoch {epoch}"):
            subgraphs = [b.to(device) for b in subgraphs]
            new_train_idx = list(range(len(input_nodes)))

            pred = model(subgraphs)
            preds[output_nodes] += pred

            torch.cuda.empty_cache()

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx], labels[val_idx]),
        evaluator(preds[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        preds,
    )
    
def get_time_elapsed(start):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2}"

def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, device):
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]

    train_batch_size = (len(train_idx) + 9) // 10
    # batch_size = len(train_idx)
    train_sampler = MultiLayerNeighborSampler([16 for _ in range(args.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(args.n_layers)
    train_dataloader = DataLoaderWrapper(
        NodeDataLoader(
            graph.cpu(),
            train_idx.cpu(),
            train_sampler,
            batch_sampler=BatchSampler(len(train_idx), batch_size=train_batch_size),
            num_workers=4,
        )
    )

    eval_sampler = MultiLayerNeighborSampler([60 for _ in range(args.n_layers)])
    # sampler = MultiLayerFullNeighborSampler(args.n_layers)
    eval_dataloader = DataLoaderWrapper(
        NodeDataLoader(
            graph.cpu(),
            torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
            eval_sampler,
            batch_sampler=BatchSampler(graph.number_of_nodes(), batch_size=32768),
            num_workers=4,
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    model = gen_model(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)

    total_time = 0
    val_score, best_val_score, final_test_score = 0, 0, 0

    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    final_pred = None
    count = 0
    start = time.time()
    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        loss = train(args, model, train_dataloader, labels, train_idx, criterion, optimizer, evaluator_wrapper, epoch, device)
        toc = time.time()
        total_time += toc - tic

        if epoch == args.n_epochs or epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                args, model, eval_dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper, epoch, device
            )

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                final_pred = pred
                count = 0
            else:
                count += 1
                if count >= args.patient and epoch >= args.min_epoch:
                    if loss > args.max_loss:
                        raise Exception("run max_loss")
                    print("patient exhausted")
                    break

            if epoch % args.log_every == 0:
                print(
                    f"\nRun: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}s"
                )
                print(
                    f"Loss: {loss:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test score: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}/{best_val_score:.4f}/{final_test_score:.4f}"
                , flush=True)

            for l, e in zip(
                [train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses],
                [train_score, val_score, test_score, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

        lr_scheduler.step(val_score)
    if loss > args.max_loss:
        raise Exception("run max_loss")
    time_elapsed = get_time_elapsed(start)
    print("*" * 50)
    print(f'Time Elapsed:  {time_elapsed}')
    print(f"Best val score: {best_val_score}, Final test score: {final_test_score}")
    print("*" * 50, flush=True)

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_score, final_test_score


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-proteins", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument('--device', type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1200, help="number of epochs")
    argparser.add_argument("--n-heads", type=int, default=8, help="number of heads")
    argparser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=6, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=64, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--type", type=str, default="DPGAT", help="GAT type")
    argparser.add_argument("--patient", type=int, default=10, help="early stopping")
    argparser.add_argument("--min_epoch", type=int, default=120, help="run at least MIN_EPOCHs")
    argparser.add_argument("--max_loss", type=float, default=0.3, help="run at least MIN_EPOCHs")
    args = argparser.parse_args()
    print(args)
    
    start = time.time()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph, labels = preprocess(graph, labels, train_idx)

    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = [], []

    i = 0
    j = 0
    while i < args.n_runs:
        seed(args.seed + i + j)
        try:
            val_score, test_score = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1, device)
        except:
            print("run failed, rerun...", flush=True)
            j += 1
            continue
        val_scores.append(val_score)
        test_scores.append(test_score)
        i += 1
    
    time_elapsed = get_time_elapsed(start)
    print(args)
    print(f"Runned {args.n_runs} times")
    print(f'Time Elapsed:  {time_elapsed}')
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
