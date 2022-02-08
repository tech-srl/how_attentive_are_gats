# How Attentive are Graph Attention Networks?

This repository is the official implementation of [How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf). 

**_January 2022_**: the paper was accepted to **ICLR'2022** !

![alt text](images/fig1.png "Figure 1 from the paper")


## Using GATv2

**GATv2 is now available as part of PyTorch Geometric library!** 
```
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
```

[https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv)

and also is [in this main directory](gatv2_conv_PyG.py).

**GATv2 is now available as part of DGL library!** 
```
from dgl.nn.pytorch import GATv2Conv
```

[https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#gatv2conv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#gatv2conv)

and also in [this repository](gatv2_conv_DGL.py).

**GATv2 is now available as part of Google's TensorFlow GNN library!** 
```
from tensorflow_gnn.graph.keras.layers.gat_v2 import GATv2Convolution
```

[https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/gnn/keras/layers/GATv2.md](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/gnn/keras/layers/GATv2.md)

## Code Structure

Since our experiments (Section 4) are based on different frameworks, this repository is divided into several sub-projects:
1. The subdirectory `arxiv_mag_products_collab_citation2_noise` contains the needed files to reproduce the results of 
Node-Prediction, Link-Prediction, and Robustness to Noise (Tables 2a, 3 and Figure 4).
2. The subdirectory `proteins` contains the needed files to reproduce the results of ogbn-proteins in Node-Prediction (Table 2b).
3. The subdirectory `dictionary_lookup` contains the need files to reproduce the results of the DictionaryLookup benchmark (Figure 3).
4. The subdirectory `tf-gnn-samples` contains the needed files to reproduce the results of the VarMisuse and QM9 datasets 
(Table 1 and Table 4).

## Requirements
Each subdirectory contains its own requirements and dependencies.

Generally, all subdirectories depend on PyTorch 1.7.1 and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) version 1.7.0 (`proteins` depends on [DGL](https://www.dgl.ai/) version 0.6.0).
The subdirectory `tf-gnn-samples` (VarMisuse and QM9) depends on TensorFlow 1.13. 

## Hardware
In general, all experiments can run on either GPU or CPU. 


## Citation
[How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf)
```
@inproceedings{
  brody2022how,
  title={How Attentive are Graph Attention Networks? },
  author={Shaked Brody and Uri Alon and Eran Yahav},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=F72ximsx7C1}
}
```







