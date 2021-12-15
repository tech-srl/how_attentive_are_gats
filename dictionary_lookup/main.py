from argparse import ArgumentParser
from attrdict import AttrDict

from experiment import Experiment
from common import Task, GNN_TYPE, STOP

def get_fake_args(
        task=Task.DICTIONARY,
        type=GNN_TYPE.GAT,
        dim=128,
        num_heads=1,
        size=10,
        num_layers=1,
        dropout=0.0,
        train_fraction=0.8,
        max_epochs=50000,
        eval_every=100,
        batch_size=1024,
        accum_grad=1,
        patience=20,
        stop=STOP.TEST_NODE,
        loader_workers=0,
        use_layer_norm=False,
        use_activation=False,
        use_residual=False,
        include_self=False,
        unroll=False,
        save=None,
        load=None,
):
    return AttrDict({
        'task': task,
        'type': type,
        'dim': dim,
        'num_heads': num_heads,
        'size': size,
        'num_layers': num_layers,
        'dropout': dropout,
        'train_fraction': train_fraction,
        # 'unseen_combs': unseen_combs,
        'max_epochs': max_epochs,
        'eval_every': eval_every,
        'batch_size': batch_size,
        'accum_grad': accum_grad,
        'stop': stop,
        'patience': patience,
        'loader_workers': loader_workers,
        'use_layer_norm': use_layer_norm,
        'use_activation': use_activation,
        'use_residual': use_residual,
        'include_self': include_self,
        'unroll': unroll,
        'save': save,
        'load': load
    })


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", dest="task", default=Task.DICTIONARY, type=Task.from_string, choices=list(Task),
                        required=False)
    parser.add_argument("--type", dest="type", default=GNN_TYPE.GAT, type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                        required=False)
    parser.add_argument("--dim", dest="dim", default=128, type=int, required=False)
    parser.add_argument("--num_heads", dest="num_heads", default=1, type=int, required=False)
    parser.add_argument("--size", dest="size", default=10, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=1, type=int, required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.0, type=float, required=False)
    parser.add_argument("--train_fraction", dest="train_fraction", default=0.8, type=float, required=False)
    # parser.add_argument('--unseen_combs', action='store_true')
    parser.add_argument("--max_epochs", dest="max_epochs", default=50000, type=int, required=False)
    parser.add_argument("--eval_every", dest="eval_every", default=100, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=1024, type=int, required=False)
    parser.add_argument("--accum_grad", dest="accum_grad", default=1, type=int, required=False)
    parser.add_argument("--stop", dest="stop", default=STOP.TEST_NODE, type=STOP.from_string, choices=list(STOP),
                        required=False)
    parser.add_argument("--save", dest="save", type=str, required=False)
    parser.add_argument("--load", dest="load", type=str, required=False)
    parser.add_argument("--plot", dest="plot", default=None, type=int, required=False, help='plots the attention for a specific example')
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)
    parser.add_argument("--loader_workers", dest="loader_workers", default=0, type=int, required=False)
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--use_activation', action='store_true')
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--include_self', action='store_true')
    parser.add_argument('--unroll', action='store_true', help='use the same weights across GNN layers')

    args = parser.parse_args()
    if args.load is None:
        Experiment(args).run()
    else:
        exp = Experiment.load_model(args.load, get_fake_args())
        test_node_acc, test_graph_acc = exp.eval()
        print(
            f'Test-node acc: {test_node_acc:.4f}, '
            f'Test-graph acc: {test_graph_acc:.4f}')
        if args.plot is not None:
            exp.plot_figures(args.plot)


