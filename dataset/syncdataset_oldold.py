import numpy as np
import networkx as nx
import random
import torch
import dgl
from .tu import DiffpoolDataset


class SyncPoolDataset():
    """A synthetic dataset for graph pooling.
    This dataset contains several subgraphs, with only sparse connections among
    them. The set of subgraphs consist of 2 types, A and B. If there are more A
    than in B, then the whole graph is classified as A; vice versa.
    Parameters
    ----------
    num_graphs: int
        Number of composite graph in this dataset
    gen_graph_type: string
        The type of graph we use to generate composite graph.
        For now, we assume both class are generated from the same graph
        generator (only the node feature is different!)
    num_sub_graphs: int
        Number of subgraphs in each component. For now we assume it's a fixed
        number, but it could change.
    feature_type: string
        feature type of each sub graph. Could be gaussian with different mean
        and variance, tuned by subgraph type.
    mode: string
        decide what to return
    Return
    ------
    1) If backend = default: return nx graph and np feature tensor.
    2) If backend = DGL: return DGL graph and DGL backend tensor.
    """

    def __init__(self, num_graphs, gen_graph_type='tu',
                 num_sub_graphs=3,
                 feature_type='gaussian', mode='train'):
        super(SyncPoolDataset, self).__init__()
        self.num_graphs = num_graphs
        self.gen_graph_type = gen_graph_type
        self.num_sub_graphs = num_sub_graphs
        self.feature_type = feature_type
        self.min_nodes = 30
        self.max_nodes = 50
        self.min_deg = 3
        self.max_deg = 5
        self.A_params = {'label': 0, 'mean': np.random.uniform(0, 1),
                         'variance': np.random.uniform(0, 1),
                         'dim': 32}
        self.B_params = {'label': 1, 'mean': np.random.uniform(0, 1),
                         'variance': np.random.uniform(0, 1),
                         'dim': 32}
        self.tu_name = 'ENZYMES'
        self.d_k_a = {}
        self.d_k_a['feature_mode'] = 'default'
        self.d_k_a['assign_feat'] = 'id'
        self.tu_dataset = DiffpoolDataset(self.tu_name, use_node_attr=True,
                                          use_node_label=False, mode=mode,
                                          train_ratio=0.8, test_rato=0.1,
                                          **self.d_k_a)
        self.graphs = []
        self.features = []
        self.labels = []
        self.n2sub_labels = []
        self.gen_graphs()

        BACKEND = 'DGL'

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (nx.to_numpy_matrix(self.graphs[idx]), self.features[idx]), self.labels[idx], \
               self.n2sub_labels[idx]

    def _get_all(self):
        return self.graphs, self.features, self.labels

    def gen_graphs(self):
        for n in range(self.num_graphs):
            graphs = []
            feats = []
            n2sub_labels = []
            split_ratio = np.random.uniform(0, 1)
            n_A = int(split_ratio * self.num_sub_graphs)
            n_B = self.num_sub_graphs - n_A
            print(f"n_A: {n_A}          n_B: {n_B}")
            for _ in range(n_A):
                if self.gen_graph_type == 'default':
                    g, feat = self.gen_component(self.feature_type, self.A_params,
                                                 self.min_nodes, self.max_nodes,
                                                 self.min_deg, self.max_deg)
                elif self.gen_graph_type == 'tu':
                    g, feat = self.gen_component_from_dataset(self.tu_dataset,
                                                              0)
                    for (key, value) in g.ndata.items():
                        g.ndata[key] = torch.Tensor(value)
                    n2sub_labels = n2sub_labels + [0 for _ in
                                                   range(g.number_of_nodes())]
                    # here we assume TU returns DGLGraph
                    # g = self.to_networkx(g).to_undirected()


                graphs.append(g)
                feats.append(feat)
            for _ in range(n_B):
                if self.gen_graph_type == 'default':
                    g, feat = self.gen_component(self.feature_type, self.B_params,
                                                 self.min_nodes, self.max_nodes,
                                                 self.min_deg, self.max_deg)
                elif self.gen_graph_type == 'tu':
                    g, feat = self.gen_component_from_dataset(self.tu_dataset,
                                                              1)
                    for (key, value) in g.ndata.items():
                        g.ndata[key] = torch.Tensor(value)
                    n2sub_labels = n2sub_labels + [1 for _ in
                                                   range(g.number_of_nodes())]
                    # g = self.to_networkx(g).to_undirected()
                graphs.append(g)
                feats.append(feat)

            if n_A > n_B:
                composite_label = 0
            else:
                composite_label = 1
            compo_g = self.connect_subgraphs(graphs)
            compo_feats = np.concatenate(feats, axis=0)
            self.graphs.append(compo_g)
            self.features.append(compo_feats)
            self.labels.append(composite_label)
            self.n2sub_labels.append(n2sub_labels)

    def connect_subgraphs(self, graph_list):
        '''
        parameters : idk
        return : a randomly connected graph.
        '''
        # we assume num sub graph is 15
        super_g = nx.erdos_renyi_graph(len(graph_list), 0.6, seed=123,
                                       directed=False)
        while (not nx.is_connected(super_g)):
            super_g = nx.erdos_renyi_graph(len(graph_list), 0.6, seed=123,
                                           directed=False)
        # first we cast the graph_list to a large graph using DGL API
        if self.gen_graph_type == 'default':
            graph_list = [self.from_networkx(g) for g in graph_list]
        batch_graph = dgl.batch(graph_list)
        bg_node_list = batch_graph.batch_num_nodes
        g = self.de_batch(batch_graph)
        accu_bg_node_list = [sum(bg_node_list[:i + 1]) for i in
                             range(len(bg_node_list))]

        for (src, dst) in super_g.edges():
            a_src = random.randint((0 if src == 0 else
                                    accu_bg_node_list[src - 1]),
                                   accu_bg_node_list[src] - 1)
            a_dst = random.randint((0 if dst == 0 else
                                    accu_bg_node_list[dst - 1]),
                                   accu_bg_node_list[dst] - 1)
            g.add_edges([a_src], [a_dst])
            # add super edge type?

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
        # dg.ndata['feat'] = g.ndata['feat']

        return dg

    def gen_component_from_dataset(self, dataset, label=0):
        # dataset statistics inquiry
        # assert required label class is actually there
        g, feat = dataset.sample(label)

        return g, feat

    def gen_component(self, feature_type, feature_params,
                      min_nodes, max_nodes, min_deg, max_deg):
        num_n = int(random.randint(min_nodes, max_nodes) / 2) * 2
        deg = int(random.randint(min_deg, max_deg) / 2) * 4
        g = nx.random_regular_graph(deg, num_n, seed=123)
        g = max(nx.connected_component_subgraphs(g), key=len)
        # ensure connected component
        if feature_type == 'gaussian':
            assert 'mean' in feature_params.keys()
            assert 'variance' in feature_params.keys()
            assert 'dim' in feature_params.keys()
            feat = np.random.normal(feature_params['mean'],
                                    feature_params['variance'],
                                    (g.number_of_nodes(),
                                     feature_params['dim']))

            # feat_dict = {i: {'feat': feat[i,:]} for i in range(feat.shape[0])}
            # nx.set_node_attributes(g, feat_dict)
        else:
            raise NotImplementedError

        return g, feat
