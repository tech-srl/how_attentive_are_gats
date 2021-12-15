import os
import torch
import torch_scatter

from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

import numpy as np
import random
from attrdict import AttrDict
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

from common import StoppingCriterion
from models.graph_model import GraphModel

class Experiment():
    def __init__(self, args):
        self.task = args.task
        gnn_type = args.type
        self.size = args.size
        num_layers = args.num_layers
        self.dim = args.dim
        self.unroll = args.unroll
        self.train_fraction = args.train_fraction
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.accum_grad = args.accum_grad
        self.eval_every = args.eval_every
        self.loader_workers = args.loader_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stopping_criterion = StoppingCriterion(args.stop)
        self.patience = args.patience
        self.save_path = args.save
        self.args = args

        seed = 11
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.X_train, self.X_test, dim0, out_dim, self.criterion = \
            self.task.get_dataset(self.size, self.train_fraction, unseen_combs=True)

        self.model = GraphModel(gnn_type=gnn_type, num_layers=num_layers, dim0=dim0, h_dim=self.dim, out_dim=out_dim,
                                unroll=args.unroll,
                                layer_norm=args.use_layer_norm,
                                use_activation=args.use_activation,
                                use_residual=args.use_residual,
                                num_heads=args.num_heads,
                                dropout=args.dropout,
                                ).to(self.device)

        print(f'Starting experiment')
        self.print_args(args)
        print(f'Training examples: {len(self.X_train)}, test examples: {len(self.X_test)}')

    def print_args(self, args):
        if type(args) is AttrDict:
            for key, value in args.items():
                print(f"{key}: {value}")
        else:
            for arg in vars(args):
                print(f"{arg}: {getattr(args, arg)}")
        print()

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', threshold_mode='abs', factor=0.5, patience=10)
        print('Starting training')

        epochs_no_improve = 0
        for epoch in range(1, (self.max_epochs // self.eval_every) + 1):
            self.model.train()
            loader = DataLoader(self.X_train * self.eval_every, batch_size=self.batch_size, shuffle=True,
                                pin_memory=True, num_workers=self.loader_workers)

            total_loss = 0
            total_num_nodes = 0
            train_per_node_correct = 0
            total_num_graphs = 0
            train_per_graph_correct = 0
            optimizer.zero_grad()
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                out, targets_batch = self.model(batch)
                loss = self.criterion(input=out, target=batch.y)
                total_num_graphs += batch.num_graphs
                total_num_nodes += targets_batch.size(0)
                total_loss += (loss.item() * targets_batch.size(0))
                _, train_per_node_pred = out.max(dim=1)
                per_node_correct = train_per_node_pred.eq(batch.y)
                train_per_node_correct += per_node_correct.sum().item()
                train_per_graph_correct += torch_scatter.scatter_min(
                    index=targets_batch, src=per_node_correct.double())[0].sum().item()

                loss = loss / self.accum_grad
                loss.backward()
                if (i + 1) % self.accum_grad == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            avg_training_loss = total_loss / total_num_nodes
            train_per_node_acc = train_per_node_correct / total_num_nodes
            train_per_graph_acc = train_per_graph_correct / total_num_graphs

            test_node_acc, test_graph_acc = self.eval()
            cur_lr = [g["lr"] for g in optimizer.param_groups]

            stopping_threshold = 0.0001
            should_stop, relevant_value = self.stopping_criterion.is_met(train_loss=avg_training_loss,
                                                                         train_node_acc=train_per_node_acc,
                                                                         train_graph_acc=train_per_graph_acc,
                                                                         test_node_acc=test_node_acc,
                                                                         test_graph_acc=test_graph_acc,
                                                                         stopping_threshold=stopping_threshold)
            if should_stop:
                self.stopping_criterion.update_best(train_node_acc=train_per_node_acc,
                                                    train_graph_acc=train_per_graph_acc,
                                                    test_node_acc=test_node_acc,
                                                    test_graph_acc=test_graph_acc,
                                                    epoch=epoch * self.eval_every)
                epochs_no_improve = 0
                new_best_str = self.stopping_criterion.new_best_str()
            else:
                epochs_no_improve += 1
                new_best_str = ''

            scheduler.step(relevant_value)
            print(
                f'Epoch {epoch * self.eval_every}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, '
                f'Train-node acc: {train_per_node_acc:.4f}, '
                f'Train-graph acc: {train_per_graph_acc:.4f}, '
                f'Test-node acc: {test_node_acc:.4f}, '
                f'Test-graph acc: {test_graph_acc:.4f} {new_best_str}')
            if relevant_value == 1.0:
                break
            if epochs_no_improve >= self.patience:
                print(
                    f'{self.patience} * {self.eval_every} epochs without {self.stopping_criterion} improvement, stopping. ')
                break

        self.stopping_criterion.print_best()
        if self.save_path is not None:
            self.save_model()

        return self.stopping_criterion

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            loader = DataLoader(self.X_test, batch_size=self.batch_size, shuffle=False,
                                pin_memory=True, num_workers=self.loader_workers)

            total_num_nodes = 0
            total_per_node_correct = 0
            total_num_graphs = 0
            total_per_graph_correct = 0

            for batch in loader:
                batch = batch.to(self.device)
                out, targets_batch = self.model(batch)
                _, pred = out.max(dim=1)

                total_num_nodes += targets_batch.size(0)
                total_num_graphs += batch.num_graphs

                total_per_node_correct += pred.eq(batch.y).sum().item()
                total_per_graph_correct += torch_scatter.scatter_min(
                    index=targets_batch, src=pred.eq(batch.y).double())[0].sum().item()

            per_node_acc = total_per_node_correct / total_num_nodes
            per_graph_acc = total_per_graph_correct / total_num_graphs
            return per_node_acc, per_graph_acc

    def save_model(self):
        print(f'Saving model to: {self.save_path}')
        p = Path(self.save_path)
        os.makedirs(p.parent, exist_ok=True)
        torch.save({'model': self.model.state_dict(),
                    'args': self.args}, self.save_path)
        print(f'Saved model')

    @staticmethod
    def complete_missing_args(args, default_args):
        for key, value in default_args.items():
            if key not in args:
                print(f"Missing key '{key}' in saved model, setting value: {value}")
                setattr(args, key, value)

        return args

    @staticmethod
    def load_model(path, default_args=None):
        print(f'Loading model from: {path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        saved_args = checkpoint['args']
        if default_args is not None:
            saved_args = Experiment.complete_missing_args(saved_args, default_args)
        exp = Experiment(saved_args)
        exp.device = device
        exp.model.load_state_dict(checkpoint['model'])
        exp.model.to(device)
        print(f'Loaded model')
        return exp

    def plot_figures(self, example_num):
        example = self.X_test[example_num].to(self.device)
        example.batch = [0 for _ in range(example.num_nodes)]

        pred, attention_per_edge, edge_index = self.model.attention_per_edge(example)
        per_node_acc = pred.eq(example.y).detach().cpu().numpy()
        print(f'Node accuracy: {per_node_acc} ({np.mean(per_node_acc):.2f})')

        attention_per_key_query = self.get_attention_per_key_and_query(attention_per_edge, edge_index)
        key_labels = [f'$k{i}$' for i in range(self.size)]

        if self.model.layers[0].add_self_loops and self.args.include_self:
            key_labels += ['self']
        query_labels = [f'$q{i}$' for i in range(self.size)]
        self.plot_heatmap_and_line(attention_per_key_query,
                                   xticklabels=key_labels,
                                   yticklabels=query_labels)
    def plot_heatmap_and_line(self, data, xticklabels='auto', yticklabels='auto'):
        if data.shape[0] > 1:
            data = np.expand_dims(data[1], axis=0)
        print('gat = np.array([')
        for row in data[0]:
            rowstr = ', '.join([f'{x:.2f}' for x in row])
            print(f'    [{rowstr}],')
        print(']')

        size = 3
        tik_label_size = 15
        fig, axes = plt.subplots(2, 1, figsize=(size+2, (size+1) * 2))
        plt.yticks(rotation=0)
        for i in range(data.shape[0]):
            cur_ax = axes[0]
            if data.shape[0] > 1:
                axes[i].set_title(f'Head #{i}')
                cur_ax = axes[i]
            ax = sb.heatmap(data[i], annot=True, fmt='.2f', cbar=False,
                            xticklabels=xticklabels, yticklabels=yticklabels, ax=cur_ax)
            ax.xaxis.tick_top()
            ax.tick_params(labelsize=tik_label_size)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        cur_ax = axes[1]
        d = {q: row for q, row in zip(yticklabels, data[i])}
        df = pd.DataFrame(d, index=xticklabels)
        ax = sb.lineplot(data=df, ax=cur_ax)
        ax.tick_params(labelsize=tik_label_size)


        plt.setp(ax.get_legend().get_texts(), fontsize=tik_label_size)
        plt.subplots_adjust(hspace=0.08, top=0.96, bottom=0.04)

        plt.legend(bbox_to_anchor=(0.99, 0.45), loc='right', prop={'size': tik_label_size, }, labelspacing=0.1,
                   borderaxespad=0.,)

        # plt.savefig('gatv2.pdf')
        plt.show()

    def get_attention_per_key_and_query(self, attention_per_edge, edge_index):
        result_y_size = self.size + 1 if (self.model.layers[0].add_self_loops and self.args.include_self) else self.size
        if len(attention_per_edge.shape) == 1:
            attention_per_edge = attention_per_edge.expand_dims(axis=-1)
        result = np.zeros(shape=(attention_per_edge.shape[-1], self.size, result_y_size))
        for head_idx in range(attention_per_edge.shape[-1]):
            for src, tgt, score in zip(edge_index[0], edge_index[1], attention_per_edge[:,head_idx]):
                if src == tgt:
                    continue
                if tgt >= self.size:
                    continue
                if src < self.size:
                    src = self.size * 2
                result[head_idx, tgt, src - self.size] = score
        return result
