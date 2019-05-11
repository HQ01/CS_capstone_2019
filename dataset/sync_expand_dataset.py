import math
import bisect
import numpy as np
import networkx as nx
import random
import torch
import dgl

class SyncExpandDataset():
    """
    Synthetic expansion of a given dataset
    """

    def __init__(self, n_graph_per_expand, perturbation='uniform',
                 base_graph_gen=None, expand_graph_type='clique'):
        super(SyncExpandDataset, self).__init__()
        random.seed(0)
        np.random.seed(0)
        self.n_graph_per_expand = n_graph_per_expand
        self.perturbation = perturbation
        self.base_graph_gen = base_graph_gen
        self.expand_graph_type = expand_graph_type
        self.graphs = []
        self.features = []
        self.labels = []
        self.n2sub_labels = []
        self.gen_graphs()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (nx.to_numpy_matrix(self.graphs[idx]), self.features[idx]), self.labels[idx],\
                self.n2sub_labels[idx]

    def gen_graphs(self):
        """
        Generate graphs
        """
        for i in range(len(self.base_graph_gen)):
            for _ in range(self.n_graph_per_expand):
                self.gen_specific_hyper_graph(self.base_graph_gen[i])

    def _generate_clique(self, min_v, max_v, perturb_base, num_v=None):
        perturb_node_feat = []
        if not num_v:
            print("random num_v for each node")
            num_v = np.random.randint(min_v, max_v)
        g = nx.complete_graph(num_v)
        for i in range(num_v):
            perturb = np.random.uniform(low=-1.0, high=1.0, size=perturb_base.shape)
            perturb_node_feat.append(perturb_base + perturb)

        # Here it's networkx graph
        return g, perturb_node_feat

    def gen_specific_hyper_graph(self, hyper_base):
        # hyper_base is actually a DGL graph with PYTORCH node features
        graphs = []
        feats = []
        n2sub_labels = []
        # hyper base consist of both graph and label

        #Future implementation of given component graph generator
        component_gen = None

        # set expand size
        min_v = 4
        max_v = 10
        num_v = np.random.randint(min_v, max_v)

        for n in hyper_base[0].nodes():
            #print(hyper_base[0].ndata['feat'])
            #print(type(hyper_base[0].ndata['feat'][0]))
            #print(hyper_base[0].ndata['feat'].shape)
            #print(hyper_base[0].ndata['feat'][0].shape)
            perturb_base = np.expand_dims(hyper_base[0].ndata['feat'][int(n)],
                                          axis=0)

            #Future implementation of given component graph generator
            component_gen = None
            if component_gen:
                pass
                #g, c_feats = self.gen_component_from_dataset()

            # networkx graph!
            graph, c_feats = self._generate_clique(min_v, max_v, perturb_base,
                                                  num_v)
            graph = self.from_networkx(graph)
            graphs.append(graph)
            feats = feats + c_feats
            n2sub_labels = n2sub_labels = [3 for _ in range(graph.number_of_nodes())]

        compo_g = self.connect_subgraphs(graphs, hyper_base)
        compo_feats = np.concatenate(feats, axis=0)

        compo_label = int(hyper_base[1])

        self.graphs.append(compo_g)
        self.features.append(compo_feats)
        self.labels.append(compo_label)
        self.n2sub_labels.append(n2sub_labels)

    def connect_subgraphs(self, graphs, hyper_base):
        batch_graph = dgl.batch(graphs)
        bg_node_list = batch_graph.batch_num_nodes
        g = self.de_batch(batch_graph)
        accu_bg_node_list = [sum(bg_node_list[:i+1]) for i
                                in range(len(bg_node_list))]

        src_list = []
        dst_list = []
        #print(list(zip(hyper_base[0].edges())))
        hyper_src, hyper_dst = hyper_base[0].edges()
        edge_pair = list(zip(list(hyper_src),list(hyper_dst)))
        #print(edge_pair)
        #raise NotImplementedError
        for (src, dst) in edge_pair:
            a_src = random.randint((0 if src == 0 else
                                    accu_bg_node_list[src-1]),
                                    accu_bg_node_list[src]-1)

            a_dst = random.randint((0 if dst == 0 else
                                    accu_bg_node_list[dst-1]),
                                    accu_bg_node_list[dst]-1)
            src_list.append(a_src)
            dst_list.append(a_dst)
        g.add_edges(src_list, dst_list)

        g = self.to_networkx(g)

        return g



    def from_networkx(self, g):
        dgl_g = dgl.DGLGraph()
        dgl_g.from_networkx(g)
        return dgl_g

    def to_networkx(self, g):
        nx_g = g.to_networkx()
        return nx_g

    def de_batch(self, g):
        dg = dgl.DGLGraph()
        dg.add_nodes(g.number_of_nodes())
        dg.add_edges(*list(g.edges()))

        return dg

