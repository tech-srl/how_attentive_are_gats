# DictionaryLookup Benchmark

This repository can be used to reproduce the experiments of 
Section 4.1 in the paper, for the "DictionaryLookup" problem. 


# The DictionaryLookup problem
![alt text](./images/fig2.png "Figure 2 from the paper")

## Requirements

### Dependencies
This project is based on PyTorch 1.7.1 and the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) library.
First, install PyTorch from the official website: [https://pytorch.org/](https://pytorch.org/).
PyTorch Geometric requires manual installation, and we thus recommend to use the instructions in  [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

The `requirements.txt` lists the additional requirements.

Eventually, run the following to verify that all dependencies are satisfied:
```setup
pip install -r requirements.txt
```

## Reproducing Experiments

To run a single experiment from the paper, run:

```
python main.py --help
```
And see the available flags.
For example, to train a GATv2 with size=10 and num_heads=1, run:
```
python main.py --task DICTIONARY --size 10 --num_heads 1 --type GAT2 --eval_every 10
```  

Alternatively, to train a GAT with size=10 and num_heads=8, run:
```
python main.py --task DICTIONARY --size 10 --num_heads 8 --type GAT --eval_every 10
```

## Experiment with other GNN types
To experiment with other GNN types:
* Add the new GNN type to the `GNN_TYPE` enum [here](common.py#L30), for example: `MY_NEW_TYPE = auto()`
* Add another `elif self is GNN_TYPE.MY_NEW_TYPE:` to instantiate the new GNN type object [here](common.py#L57)
* Use the new type as a flag for the `main.py` file:
```
python main.py --type MY_NEW_TYPE ...
```

