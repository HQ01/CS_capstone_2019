import argparse
import atexit
import torch.nn as nn
import mlflow

from batched_train import Controller
from model.layers.diffpool import BatchedDiffPool
from model.layers.graphsage import BatchedGraphSAGE
from utils import del_tensorboard


class Layer:
    def __init__(self, dataset='ENZYMES', hidden_size=30):
        self.hidden_size = hidden_size
        self.dataset = dataset

    def __call__(self, input_shape, output_shape, pool_size, link_pred, ncut, gumbel, rtn):
        hidden_size = self.hidden_size
        if self.dataset == 'ENZYMES':
            model = nn.ModuleList([
                BatchedGraphSAGE(input_shape, hidden_size),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedDiffPool(hidden_size, pool_size, hidden_size,
                                link_pred=link_pred, ncut=ncut, gumbel=gumbel, rtn=rtn),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedGraphSAGE(hidden_size, output_shape),
            ])
        elif self.dataset == 'DD':
            model = nn.ModuleList([
                BatchedGraphSAGE(input_shape, hidden_size),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedDiffPool(hidden_size, pool_size, hidden_size,
                                link_pred=link_pred, ncut=ncut, gumbel=gumbel, rtn=rtn),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedDiffPool(hidden_size, int(pool_size * 0.5), hidden_size,
                                link_pred=link_pred, ncut=ncut, gumbel=gumbel, rtn=rtn),
                BatchedGraphSAGE(hidden_size, output_shape),
            ])

        elif self.dataset.lower().startswith('sync'):
            model = nn.ModuleList([
                BatchedGraphSAGE(input_shape, hidden_size),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedDiffPool(hidden_size, pool_size, hidden_size,
                                link_pred=link_pred, ncut=ncut, gumbel=gumbel,
                                rtn=rtn),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedGraphSAGE(hidden_size, output_shape),
            ])
            
        else:
            model = nn.ModuleList([
                BatchedGraphSAGE(input_shape, hidden_size),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedDiffPool(hidden_size, pool_size, hidden_size,
                                link_pred=link_pred, ncut=ncut, gumbel=gumbel, rtn=rtn),
                BatchedGraphSAGE(hidden_size, hidden_size),
                BatchedDiffPool(hidden_size, int(pool_size * 0.5), hidden_size,
                                link_pred=link_pred, ncut=ncut, gumbel=gumbel, rtn=rtn),
                BatchedGraphSAGE(hidden_size, output_shape),
            ])
        return model


base_config = {'no_cuda': False,
               'epochs': 3000,
               'link_pred': True,
               'dataset': 'Synthetic',
               'batch_size': 40,
               'train_ratio': 0.9,
               'pool_ratio': 0.5,
               'hidden_size': 30,
               'ncut': False,
               'log_interval': 20,
               'gumbel': False,
               'scheduler': False,
               'rtn': -1}

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                     help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
parser.add_argument('--link-pred', action='store_true', default=None, help='Enable Link Prediction Loss')
parser.add_argument('--ncut', action='store_true', default=None, help='Enable Link Prediction Loss')
parser.add_argument('--prod', action='store_true', default=False)
parser.add_argument('--dataset', help="Choose dataset: ENZYMES, DD, Synthetic")
parser.add_argument('--batch-size', type=int, help="Batch Size")
parser.add_argument('--pool-size', type=int, default=None, help="Pool Size")
parser.add_argument('--scheduler', action='store_true', default=False, help="Scheduler")
# parser.add_argument('--train-ratio', default=0.9, type=float, help="Train/Val split ratio")
parser.add_argument('--pool-ratio', type=float, help="Pool ratio")
# parser.add_argument('--log-interval', default=20, type=int, help="Log Interval")
parser.add_argument('--gumbel', action='store_true', default=False, help='Use gumbel softmax')
parser.add_argument('--viz', action='store_true', default=True)
parser.add_argument('--hidden-size', type=int)
parser.add_argument('--clip', action='store_true', default=True)
parser.add_argument('--logname', default='NEW_Diffpool_SYNTHETIC-H_globalhop_ret1e-1_10step_1e-2_wrefine_wlp_woentro_3000step', type=str)
parser.add_argument('--rtn', type=int, help="routing num")
args = parser.parse_args()

print(vars(args))
args_dict = vars(args)
for key, value in args_dict.copy().items():
    if value is None:
        args_dict.pop(key)

base_config.update(args_dict)
base_config.update({
    'layers': Layer(base_config['dataset'], base_config['hidden_size'])
})
print("############--------################")
print(base_config)

if not base_config.get('prod', False):
    atexit.register(del_tensorboard)

print("############--------################")

controller = Controller(base_config)

with mlflow.start_run():
    controller.main()
