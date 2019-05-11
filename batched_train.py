from __future__ import division
from __future__ import print_function

import argparse
import pickle
import random


import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np
import utils
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split

from MNIST import GraphTransform
from batched_model import BatchedModel
from model.layers.diffpool import BatchedDiffPool
from dataset.dataset import TUDataset, CollateFn
from dataset.syncdataset import SyncPoolDataset
from dataset.dummydataset import DummyClique
from dataset.tu import DiffpoolDataset
from dataset.sync_expand_dataset import SyncExpandDataset
import matplotlib.pyplot as plt
import seaborn
import networkx as nx

# from model.routing_model import BatchedRoutingModel
import mlflow


class Controller:
    def __init__(self, args):
        self.args = args
        #torch.manual_seed(0)

    def main(self):
        args = self.args
        device = "cuda" if not args['no_cuda'] and torch.cuda.is_available() else "cpu"

        ################## Setup Dataset ##################################
        if args['dataset'].lower().startswith("syn"):
            # Manual toggle of different datasets:
            #d_k_a = {'feature_mode': 'default', 'assign_feat': 'id'}
            #enzyme_gen = DiffpoolDataset('ENZYMES', use_node_attr=True,
            #                             use_node_label=False,
            #                             mode='train',
            #                             train_ratio=0.8,
            #                             test_ratio=0.1,
            #                             **d_k_a)
            #clique_gen = DummyClique(600, [10, 20, 30, 40, 50], [20, 30, 40, 50, 60],
            #                         5)
            #dataset = SyncPoolDataset(600, graph_dataset=clique_gen,
            #                          num_sub_graphs=10, mode='train')

            # Fix a random dataset
            pickle_in = open('sync_fix_H.pickle', 'rb')
            #pickle_in = open('sync_fix.pickle', 'rb')
            dataset = pickle.load(pickle_in)
            pickle_in = open('sync_fix_H_val.pickle', 'rb')
            dataset_val = pickle.load(pickle_in)
            pickle_in = open('sync_fix_H_test.pickle', 'rb')
            dataset_test = pickle.load(pickle_in)

            # Hijack node feature
            adversial_feature = []
            for i in range(len(dataset.features)):
                feat = np.ones(dataset.features[i].shape)
                adversial_feature.append(feat)
            max_num_nodes_candidate = []
            for ds in (dataset, dataset_val, dataset_test):
                max_num_nodes = max(np.array([item[0][0].shape[0] for item in ds]))
                max_num_nodes_candidate.append(max_num_nodes)
            max_num_nodes = max(max_num_nodes_candidate)
            #dataset.features = adversial_feature
            #print('manually set node features to constant')
        else:
            dataset = TUDataset(args['dataset'])
            max_num_nodes = max(np.array([item[0][0].shape[0] for item in dataset]))
        
        # Turn this off if dataset not pre_separated
        # \TODO move this to argparser
        pre_separated = True


        dataset_size = len(dataset)
        train_size = int(dataset_size * args['train_ratio'])
        val_size = dataset_size - train_size
        mean_num_nodes = int(np.array([item[0][0].shape[0] for item in dataset]).mean())
        n_classes = int(max([item[1] for item in dataset])) + 1
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        labels = np.array(list(zip(*dataset))[1])
        for train_index, test_index in skf.split(np.zeros_like(labels), labels):
            # THIS K-FOLD IS BROKEN
            #\TODO Fix
            utils.writer = utils.CustomLogger(comment='|'.join([args['dataset'], args['logname']]))
            utils.writer.add_text("args", str(args))
            utils.writer.log_args(args)
            utils.writer.log_py_file()


            train_val_data = torch.utils.data.Subset(dataset, train_index)
            train_val_labels = np.array(list(zip(*train_val_data))[1])
            train_data, val_data = train_test_split(train_val_data, train_size=0.8,
                                                    stratify=train_val_labels)
            test_data = torch.utils.data.Subset(dataset, test_index)
            viz_data = torch.utils.data.Subset(test_data, [0])

            input_shape = int(dataset[0][0][1].shape[-1])
            if pre_separated:
                #\TODO not k-fold ready yet!
                train_loader = DataLoader(dataset, batch_size=args['batch_size'],
                                            shuffle=True,
                                            collate_fn=CollateFn(max_num_nodes, device))
                val_loader = DataLoader(dataset_val, batch_size=args['batch_size'],
                                            shuffle=True,
                                            collate_fn=CollateFn(max_num_nodes, device))
                test_loader = DataLoader(dataset_test, batch_size=args['batch_size'],
                                            shuffle=True,
                                            collate_fn=CollateFn(max_num_nodes, device))
            
            else:
                train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True,
                                        collate_fn=CollateFn(max_num_nodes, device))
                val_loader = DataLoader(val_data, batch_size=args['batch_size'], shuffle=False,
                                        collate_fn=CollateFn(max_num_nodes, device))
                test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False,
                                        collate_fn=CollateFn(max_num_nodes, device))

            viz_loader = DataLoader(viz_data, batch_size=1,
                                    collate_fn=CollateFn(max_num_nodes, device))
            ############### Record Config and setup model ###############################
            config = {}
            config.update(args)
            if args.get('pool_size', None) is not None:
                pool_size = args['pool_size']
            else:
                pool_size = int(mean_num_nodes * args['pool_ratio'])
            for k, v in locals().copy().items():
                if k in ['device', 'args', 'pool_size', 'input_shape', 'n_classes']:
                    config[k] = v
            config['rtn'] = args['rtn']
            config['link_pred'] = args['link_pred']
            config['min_cut'] = True
            print("############################")
            print(config)
            if args['rtn'] > 0:
                tqdm.write("Using Routing Model")
            else:
                tqdm.write("Using DiffPool")
            model = BatchedModel(**config).to(device)
            print(model)
            ############### Optimizer and Scheduler #################################
            self.optimizer = optim.Adam(model.parameters())
            if config.get("scheduler", False):
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            else:
                self.scheduler = utils.ConstantScheduler()
            self.vg = True

            for e in tqdm(range(args['epochs'])):
                utils.e = e
                if args['viz']:
                    self.visualize(e, model, viz_loader)
                self.train(args, e, model, train_loader)
                self.val(args, e, model, val_loader)
                self.test(args, e, model, test_loader)
                utils.writer.log_epochs(e)

            utils.writer.log_tfboard()
            break

    def train(self, args, e, model, train_loader):
        model.train()
        result_list = []
        for i, (adj, features, masks, batch_labels, _) in enumerate(train_loader):
            utils.train_iter += 1
            graph_feat = model(features, adj, masks)
            output = model.classifier(graph_feat)
            loss = model.loss(output, batch_labels, train=True)

            loss.backward()
            if self.args['clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            self.scheduler.step(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            iter_true_sample = output.argmax(dim=1).long() == batch_labels.long()
            result_list.append(iter_true_sample)
            iter_acc = iter_true_sample.float().mean().item()
            utils.writer.add_scalar("iter acc/iter train acc", iter_acc, utils.train_iter)
            utils.writer.add_scalar("iter loss/iter train loss", loss.item(), utils.train_iter)
            if utils.train_iter % args['log_interval'] == 0:
                tqdm.write(f"{utils.train_iter} iter train acc: {iter_acc}")
        acc = torch.cat(result_list).float().mean().item()
        utils.writer.add_scalar("Acc/Epoch Train Acc", acc, e)
        tqdm.write(f"Epoch:{e}  \t train_acc:{acc:.4f}")

    def val(self, args, e, model, val_loader):
        model.eval()
        result_list = []
        # pred_list = []
        with torch.no_grad():
            for i, (adj, features, masks, batch_labels, _) in enumerate(val_loader):
                utils.val_iter += 1
                graph_feat = model(features, adj, masks)
                output = model.classifier(graph_feat)
                loss = model.loss(output, batch_labels)
                iter_true_sample = output.argmax(dim=1).long() == batch_labels.long()
                # pred_list.append(output.argmax(dim=1))
                result_list.append(iter_true_sample)
                iter_acc = iter_true_sample.float().mean().item()
                utils.writer.add_scalar("iter acc/iter val acc", iter_acc, utils.val_iter)
                utils.writer.add_scalar("iter loss/iter val loss", loss.item(), utils.val_iter)
                if utils.val_iter % args['log_interval'] == 0:
                    tqdm.write(f"{utils.val_iter} iter val acc: {iter_acc}")
        acc = torch.cat(result_list).float().mean().item()

        tqdm.write(f"Val len: {torch.cat(result_list).shape}")
        utils.writer.add_scalar("Acc/Epoch Val Acc", acc, e)
        tqdm.write(f"Epoch:{e}  \t val_acc:{acc:.4f}")
        # return torch.cat(result_list), torch.cat(pred_list)

    def test(self, args, e, model, test_loader):
        model.eval()
        result_list = []
        with torch.no_grad():
            for i, (adj, features, masks, batch_labels, _) in enumerate(test_loader):
                utils.test_iter += 1
                graph_feat = model(features, adj, masks)
                output = model.classifier(graph_feat)
                loss = model.loss(output, batch_labels)
                iter_true_sample = output.argmax(dim=1).long() == batch_labels.long()
                result_list.append(iter_true_sample)
                iter_acc = iter_true_sample.float().mean().item()
                utils.writer.add_scalar("iter acc/iter test acc", iter_acc, utils.test_iter)
                utils.writer.add_scalar("iter loss/iter test loss", loss.item(), utils.test_iter)
                if utils.test_iter % args['log_interval'] == 0:
                    tqdm.write(f"{utils.test_iter} iter test acc: {iter_acc}")
        acc = torch.cat(result_list).float().mean().item()
        tqdm.write(f"Test len: {torch.cat(result_list).shape}")
        utils.writer.add_scalar("Acc/Epoch Test Acc", acc, e)
        tqdm.write(f"Epoch:{e}  \t test_acc:{acc:.4f}")
        return acc

    def visualize(self, e, model, viz_loader):
        # Visualize
        model.eval()

        with torch.no_grad():
            for _, (adj, features, masks, batch_labels, aux) in enumerate(viz_loader):
                num_real_nodes = int(torch.sum(masks))
                adj_viz = adj[:, :num_real_nodes, :num_real_nodes]
                graph_feat = model(features, adj, masks, log=True)
                output = model.classifier(graph_feat)
                loss = model.loss(output, batch_labels)
                if self.vg:
                    self.vg = False
                    self.g: nx.DiGraph = nx.convert_matrix.from_numpy_matrix(
                        adj_viz.cpu().numpy().squeeze())
                    self.spring_layout = nx.spring_layout(self.g)
                    self.spectral_layout = nx.spectral_layout(self.g)
                if self.args['dataset'] == 'Synthetic':
                    fig, ax = plt.subplots()
                    labels = aux
                    nx.draw_networkx(self.g, pos=self.spring_layout,
                                     with_labels=False,
                                     # labels=dict(zip(range(g.number_of_nodes()), labels)),
                                     # node_color=labels,
                                     ax=ax)
                    utils.writer.add_figure("Spring/Ground Truth Graph Spring", fig, e)

                    fig, ax = plt.subplots()
                    nx.draw_networkx(self.g, pos=self.spectral_layout,
                                     with_labels=False,
                                     # labels=dict(zip(range(g.number_of_nodes()), labels)),
                                     # node_color=labels,
                                     ax=ax)
                    utils.writer.add_figure("Spectral/Ground Truth Graph Spectral", fig, e)

                for layer in model.layers:
                    if isinstance(layer, BatchedDiffPool):
                        for i, key in enumerate(
                                filter(lambda x: x.startswith("s"), layer.log.keys())):
                            fig, ax = plt.subplots()
                            seaborn.heatmap(layer.log[key].squeeze(), vmin=0, vmax=1, ax=ax)
                            utils.writer.add_figure(f"assignment matrix {i}", fig, e)
                        fig, ax = plt.subplots()
                        seaborn.heatmap(layer.log['a'].squeeze(), ax=ax)
                        utils.writer.add_figure("Adj/adjacency matrix", fig, e)

                        fig, ax = plt.subplots()
                        labels = layer.log[key].argmax(-1).squeeze().tolist()[:num_real_nodes]
                        nx.draw_networkx(self.g, pos=self.spring_layout,
                                         with_labels=False,
                                         # labels=dict(zip(range(g.number_of_nodes()), labels)),
                                         node_color=labels,
                                         ax=ax)
                        utils.writer.add_figure("Spring/Graph Spring", fig, e)
                        fig, ax = plt.subplots()
                        nx.draw_networkx(self.g, pos=self.spectral_layout,
                                         with_labels=False,
                                         # labels=dict(zip(range(g.number_of_nodes()), labels)),
                                         node_color=labels,
                                         ax=ax)
                        utils.writer.add_figure("Spectral/Graph Spectral", fig, e)
                        break

# def main():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='Disables CUDA training.')
#     parser.add_argument('--epochs', type=int, default=700,
#                         help='Number of epochs to train.')
#     parser.add_argument('--link-pred', action='store_true', default=True,
#                         help='Enable Link Prediction Loss')
#     parser.add_argument('--dataset', default='ENZYMES', help="Choose dataset: ENZYMES, DD")
#     parser.add_argument('--batch-size', default=20, type=int, help="Batch Size")
#     parser.add_argument('--pool-size', type=int, default=None, help="Pool Size")
#     parser.add_argument('--train-ratio', default=0.9, type=float, help="Train/Val split ratio")
#     parser.add_argument('--pool-ratio', default=0.6, type=float, help="Pool ratio")
#     parser.add_argument('--log-interval', default=20, type=int, help="Log Interval")
#     parser.add_argument('--gumbel', default=False, action='store_true', help='Use gumbel softmax')
#     parser.add_argument('--rtn', default=-1, type=int, help="routing num")
#     args = parser.parse_args()
#     print(vars(args))
#
#     controller = Controller(vars(args))
#     # mlflow.set_tracking_uri("")
#     with mlflow.start_run():
#         controller.main()


# if __name__ == '__main__':
#     main()
