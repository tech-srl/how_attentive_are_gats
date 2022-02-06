from random import random, sample
import torch
from tqdm import tqdm

def add_edge_noise(edge_index, num_nodes, p):
    edge_set = set(map(tuple, edge_index.transpose(0, 1).tolist()))
    num_of_new_edge = int((edge_index.size(1) // 2) * p)
    to_add = list()
    new_edges = sample(range(1, num_nodes**2 + 1), num_of_new_edge + len(edge_set) + num_nodes)
    c = 0
    for i in new_edges:
        if c >= num_of_new_edge:
            break
        s = ((i - 1) // num_nodes) + 1
        t = i - (s - 1) * num_nodes
        s -= 1
        t -= 1
        if s != t and (s, t) not in edge_set:
            c += 1
            to_add.append([s, t])
            to_add.append([t, s])
            edge_set.add((s, t))
            edge_set.add((t, s))
    print(f"num of added edges: {len(to_add)}")
    new_edge_index = torch.cat([edge_index.to('cpu'), torch.LongTensor(to_add).transpose(0, 1)], dim=1)
    return new_edge_index

