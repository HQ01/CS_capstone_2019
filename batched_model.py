import torch.nn as nn

from model.layers.diffpool import BatchedDiffPool
from model.layers.graphsage import BatchedGraphSAGE
import utils


class BatchedModel(nn.Module):
    def __init__(self, pool_size, device, input_shape, n_classes, link_pred=False, min_cut=False,
                 ncut=False, gumbel=False, layers=None, rtn=-1, **kwargs):
        super().__init__()
        if gumbel:
            print("Using Gumbel Softmax")

        self.min_cut = min_cut
        self.input_shape = input_shape
        self.link_pred = link_pred
        self.device = device
        self.hidden_size = 30
        self.output_shape = 30
        # if layers is None:
        #     self.layers = nn.ModuleList([
        #         BatchedGraphSAGE(input_shape, self.hidden_size),
        #         BatchedGraphSAGE(self.hidden_size, self.hidden_size),
        #         BatchedDiffPool(self.hidden_size, pool_size, self.hidden_size,
        #                         link_pred=link_pred, ncut=ncut, gumbel=gumbel, rtn=rtn),
        #         BatchedGraphSAGE(self.hidden_size, self.hidden_size),
        #         BatchedGraphSAGE(self.hidden_size, self.output_shape),
        #         # BatchedDiffPool(30, 1, 30)
        #     ])
        # else:
        self.layers = layers(input_shape=input_shape, output_shape=self.output_shape,
                             pool_size=pool_size, link_pred=link_pred, ncut=ncut, gumbel=gumbel,
                             rtn=rtn)
        self.classifier = nn.Sequential(nn.Linear(self.output_shape, 20),
                                        nn.ReLU(),
                                        nn.Linear(20, n_classes))
        utils.writer.add_text("Model",
                              repr(self).replace("  ", "&nbsp;&nbsp;").replace("\n", "  \n"))

    def forward(self, x, adj, mask, log=False, sparse=False):
        for layer in self.layers:
            if isinstance(layer, BatchedGraphSAGE):
                if mask.shape[1] == x.shape[1]:
                    x = layer(x, adj, mask)
                else:
                    x = layer(x, adj)
            elif isinstance(layer, BatchedDiffPool):
                # TODO: Fix if condition
                if mask.shape[1] == x.shape[1]:
                    x, adj = layer(x, adj, mask, log=log)
                else:
                    x, adj = layer(x, adj, log=log)

        # x = x * mask
        readout_x = x.sum(dim=1)
        return readout_x

    def loss(self, output, labels, train=False):
        iter_count = utils.train_iter if train else utils.val_iter
        log_name = 'train' if train else 'val'
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        utils.writer.add_scalar(f"{log_name} loss/NLL Loss", loss.item(), iter_count)
        for layer in self.layers:
            if isinstance(layer, BatchedDiffPool):
                for key, value in layer.loss_log.items():
                    utils.writer.add_scalar(f"{log_name} loss/{key}", value, iter_count)
                    loss += value
                layer.loss_log.clear()
        return loss
