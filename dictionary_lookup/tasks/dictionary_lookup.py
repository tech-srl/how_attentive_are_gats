import random

import numpy as np
import itertools
import math
import torch
import torch_geometric

from torch.nn import functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

import common


class DictionaryLookupDataset(object):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.edges, self.empty_id = self.init_edges()
        self.criterion = F.cross_entropy


    def init_edges(self):
        targets = range(0, self.size)
        sources = range(self.size, self.size * 2)
        next_unused_id = self.size
        all_pairs = itertools.product(sources, targets)
        edges = [list(i) for i in zip(*all_pairs)]

        return edges, next_unused_id

    def create_empty_graph(self, add_self_loops=False):
        edge_index = torch.tensor(self.edges, requires_grad=False, dtype=torch.long)
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, )
        return edge_index

    def get_combinations(self):
        # returns: an iterable of [permutation(size)]
        # number of combinations: size!

        max_examples = 32000 # starting to affect from size=8, because 8!==40320

        if math.factorial(self.size) > max_examples:
            permutations = [np.random.permutation(range(self.size)) for _ in range(max_examples)]
        else:
            permutations = itertools.permutations(range(self.size))

        return permutations

    def generate_data(self, train_fraction, unseen_combs):
        data_list = []

        for perm in self.get_combinations():
            edge_index = self.create_empty_graph(add_self_loops=False)
            nodes = torch.tensor(self.get_nodes_features(perm), dtype=torch.long, requires_grad=False)
            target_mask = torch.tensor([True] * (self.size) + [False] * self.size,
                                        dtype=torch.bool, requires_grad=False)
            labels = torch.tensor(perm, dtype=torch.long, requires_grad=False)
            data_list.append(Data(x=nodes, edge_index=edge_index, target_mask=target_mask, y=labels))

        dim0, out_dim = self.get_dims()
        if unseen_combs:
            X_train, X_test = self.unseen_combs_train_test_split(data_list, train_fraction=train_fraction, shuffle=True)
        else:
            X_train, X_test = train_test_split(data_list, train_size=train_fraction, shuffle=True)

        return X_train, X_test, dim0, out_dim, self.criterion

    def get_nodes_features(self, perm):
        # perm: a list of indices

        # The first row contains (key, empty_id)
        # The second row contains (key, value) where the order of values is according to perm
        nodes = [(key, self.empty_id) for key in range(self.size)]

        for key, val in zip(range(self.size), perm):
            nodes.append((key, val))

        return nodes

    def get_dims(self):
        # get input and output dims
        in_dim = self.size + 1
        out_dim = self.size
        return in_dim, out_dim

    def unseen_combs_train_test_split(self, data_list, train_fraction, shuffle=True):
        per_position_fraction = train_fraction ** (1 / self.size)
        num_training_pairs = int(per_position_fraction * (self.size ** 2))
        allowed_positions = set(random.sample(
            list(itertools.product(range(self.size), range(self.size))), num_training_pairs))
        train = []
        test = []
        for example in data_list:
            if all([(i, label.item()) in allowed_positions for i, label in enumerate(example.y)]):
                train.append(example)
            else:
                test.append(example)
        if shuffle:
            random.shuffle(train)
        return train, test

