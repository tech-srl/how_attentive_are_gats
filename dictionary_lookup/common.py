from enum import Enum, auto

from tasks.dictionary_lookup import DictionaryLookupDataset
from gnns.gat2 import GAT2Conv

from torch import nn
from torch_geometric.nn import GCNConv, GINConv, GATConv

class Task(Enum):
    DICTIONARY = auto()

    @staticmethod
    def from_string(s):
        try:
            return Task[s]
        except KeyError:
            raise ValueError()

    def get_dataset(self, size, train_fraction, unseen_combs):
        if self is Task.DICTIONARY:
            dataset = DictionaryLookupDataset(size)
        else:
            dataset = None

        return dataset.generate_data(train_fraction, unseen_combs)


class GNN_TYPE(Enum):
    GCN = auto()
    GIN = auto()
    GAT = auto()
    GATv2 = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim, num_heads):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim)
        elif self is GNN_TYPE.GIN:
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads, add_self_loops=True)
        elif self is GNN_TYPE.GATv2:
            return GAT2Conv(in_dim, out_dim // num_heads, heads=num_heads, bias=False, share_weights=True, add_self_loops=False)



class StoppingCriterion(object):
    def __init__(self, stop):
        self.stop = stop
        self.best_train_loss = -float('inf')
        self.best_train_node_acc = 0
        self.best_train_graph_acc = 0
        self.best_test_node_acc = 0
        self.best_test_graph_acc = 0
        self.best_epoch = 0

        self.name = stop.name

    def new_best_str(self):
        return f' (new best {self.name})'

    def is_met(self, train_loss, train_node_acc, train_graph_acc, test_node_acc, test_graph_acc, stopping_threshold):
        if self.stop is STOP.TRAIN_NODE:
            new_value = train_node_acc
            old_value = self.best_train_node_acc
        elif self.stop is STOP.TRAIN_GRAPH:
            new_value = train_graph_acc
            old_value = self.best_train_graph_acc
        elif self.stop is STOP.TEST_NODE:
            new_value = test_node_acc
            old_value = self.best_test_node_acc
        elif self.stop is STOP.TEST_GRAPH:
            new_value = test_graph_acc
            old_value = self.best_test_graph_acc
        elif self.stop is STOP.TRAIN_LOSS:
            new_value = -train_loss
            old_value = self.best_train_loss
        else:
            raise ValueError

        return new_value > (old_value + stopping_threshold), new_value

    def __repr__(self):
        return str(self.stop)

    def update_best(self, train_node_acc, train_graph_acc, test_node_acc, test_graph_acc, epoch):
        self.best_train_node_acc = train_node_acc
        self.best_train_graph_acc = train_graph_acc
        self.best_test_node_acc = test_node_acc
        self.best_test_graph_acc = test_graph_acc
        self.best_epoch = epoch

    def print_best(self):
        print(f'Best epoch: {self.best_epoch}')
        print(f'Best train node acc: {self.best_train_node_acc}')
        print(f'Best train graph acc: {self.best_train_graph_acc}')
        print(f'Best test node acc: {self.best_test_node_acc}')
        print(f'Best test graph acc: {self.best_test_graph_acc}')


class STOP(Enum):
    TRAIN_NODE = auto()
    TRAIN_GRAPH = auto()
    TEST_NODE = auto()
    TEST_GRAPH = auto()
    TRAIN_LOSS = auto()

    @staticmethod
    def from_string(s):
        try:
            return STOP[s]
        except KeyError:
            raise ValueError()
