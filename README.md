# How Attentive are Graph Attention Networks?

This repository is the official implementation of [How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf). 

![alt text](images/fig1.png "Figure 1 from the paper")


**The GATv2 layer is now available as part of PyTorch Geometric library!** 

[https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv)

and also is [in this main directory](gatv2_conv_PyG.py).

DGL implementation can be found [here](gatv2_conv_DGL.py) as well.

The rest of the code for reproducing the experiments in the paper will be made publicly available.

# Citation
[How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf)
```
@article{brody2021attentive,
  title={How Attentive are Graph Attention Networks?},
  author={Brody, Shaked and Alon, Uri and Yahav, Eran},
  journal={arXiv preprint arXiv:2105.14491},
  year={2021}
}
```







