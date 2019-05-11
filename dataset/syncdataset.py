import math
import bisect
import numpy as np
import networkx as nx
import random
import torch
import dgl

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
    label_type: string
        majority/config
    mode: string
        decide what to return


    Return
    ------
    1) If backend = default: return nx graph and np feature tensor.
    2) If backend = DGL: return DGL graph and DGL backend tensor.
    """

    def __init__(self, num_graphs,
                 graph_dataset=None,
                 num_sub_graphs=3,
                 default_gen_param={'feature': 'gaussian',
                                    'mean': [0, 0, 0],
                                    'variance': [0.1, 0.1, 0.1],
                                    'link_prob': [0.3, 0.6, 0.9],
                                    'dim': [32, 32, 32]},
                 mode='train',
                 num_classes=3, #\TODO assert num_classes == link_prob
                 hyper_gen_param={'ratio': [[0.2, 0.6, 0.2], [0.6, 0.2, 0.2],
                                            [0.2, 0.2, 0.6]],
                                  'label_type': 'config'}):
                 #hyper_gen_param={'ratio':[[0.8,0.2], [0.2,0.8]],
                 #                 'label_type': 'config'}):
        super(SyncPoolDataset, self).__init__()
        random.seed(0)
        np.random.seed(0)
        assert num_classes == len(hyper_gen_param['ratio'])
        self.num_graphs = num_graphs
        self.graph_dataset = graph_dataset
        self.num_sub_graphs = num_sub_graphs
        assert default_gen_param, 'default graph generator not set'
        self.default_gen_param = default_gen_param
        self.feature_type = default_gen_param['feature']
        self.label_type = hyper_gen_param['label_type']
        self.mode = mode
        self.num_classes = num_classes
        self.hyper_gen_param = hyper_gen_param
        # \TODO eliminate magic number
        self.min_nodes = 30
        self.max_nodes = 50
        self.min_deg = 3
        self.max_deg = 5
        # \TODO fix random graph gen to k classes, move graph param to input
        #self.A_params = {'label':0,'mean':np.random.uniform(0, 1),
        #                 'variance':np.random.uniform(0, 1),
        #                 'dim':32}
        #self.B_params = {'label':1, 'mean':np.random.uniform(0, 1),
        #                 'variance':np.random.uniform(0, 1),
        #                 'dim':32}
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
        #return self.graphs[idx], self.features[idx], self.labels[idx],\
        #       self.n2sub_labels[idx]

    def _get_all(self):
        return self.graphs, self.features, self.labels

    def gen_graphs(self):
        """
        Generate graphs. We need grap_gen, hyper_gen, label_type all defined.
        """
        # evenly cut the dataset.
        #print("num graph is ", self.num_graphs)
        #print("ratio is ", self.hyper_gen_param['ratio'])
        assert self.num_graphs % len(self.hyper_gen_param['ratio']) == 0, 'hyper class class not even'
        num_each_class = int(self.num_graphs /
                             len(self.hyper_gen_param['ratio']))
        for g_class in range(self.num_classes):
            print("gen graph type {}".format(g_class))
            for _ in range(num_each_class):
                # pass the graph type index as g_class
                self.gen_specific_hyper_graph(g_class)
                # take self.hyper_gen_param

    def gen_specific_hyper_graph(self, g_class):
        graphs = []
        feats = []
        n2sub_labels = []
        assert self.num_classes > 0, 'number of classes not set!'
        # random ratio should be moved outside.
        if not self.hyper_gen_param['ratio']:
            print("warning: auto-generating split ratio")
            split = [np.random.uniform(0, 1) for _ in
                     range(self.num_classes)]
            split_ratio = split/sum(split)
        else:
            split_ratio = self.hyper_gen_param['ratio'][g_class]
        print('split_ratio is', split_ratio)
        graph_num_per_class = [round(ratio*self.num_sub_graphs) for ratio in
                               split_ratio]
        assert sum(graph_num_per_class) == self.num_sub_graphs, 'wrong sum!'
        # i is the label
        for i, num in enumerate(graph_num_per_class):
            for _ in range(num):
                if not self.graph_dataset:
                    assert self.default_gen_param['mean'][i], "mean not exist"
                    default_graph_params = {'label':i,
                                            'mean':self.default_gen_param['mean'][i],
                                            'variance':self.default_gen_param['variance'][i],
                                            'dim':self.default_gen_param['dim'][i]}

                    g, feat = self.gen_component(self.feature_type,
                                                 default_graph_params,
                                                 self.min_nodes,
                                                 self.max_nodes,
                                                 self.min_deg,
                                                 self.max_deg)
                        # \TODO factor in min and max.
                else:
                    g, feat = self.gen_component_from_dataset(self.graph_dataset,i)
                    for (key,value) in g.ndata.items():
                        g.ndata[key] = torch.Tensor(value) #\TODO fixdata
                    n2sub_labels = n2sub_labels + [i for _ in
                                                   range(g.number_of_nodes())]
                graphs.append(g)
                feats.append(feat)

        compo_g = self.connect_subgraphs(graphs, g_class, graph_num_per_class)
        compo_feats = np.concatenate(feats, axis=0)
        composite_label = self.gen_label(split_ratio, g_class)
        self.graphs.append(compo_g)
        self.features.append(compo_feats)
        self.labels.append(composite_label)
        assert len(n2sub_labels) == compo_g.number_of_nodes()
        print("assertion done")
        self.n2sub_labels.append(n2sub_labels)


    def connect_subgraphs(self, graph_list, g_class, graph_num_per_class):
        '''
        parameters : idk
        return : a randomly connected graph.
        graph_list is tied to default_gen_ratio
        '''
        # we assume num sub graph is 15
        super_g = self.gen_connect_hyper_with_config(g_class,
                                                     graph_num_per_class)

        while (not nx.is_connected(super_g)):
            super_g = self.gen_connect_hyper_with_config(g_class,
                                                         graph_num_per_class)

        # first we cast the graph_list to a large graph using DGL API
        if not self.graph_dataset:
            graph_list = [self.from_networkx(g) for g in graph_list]
        batch_graph = dgl.batch(graph_list)
        bg_node_list = batch_graph.batch_num_nodes
        g = self.de_batch(batch_graph)
        accu_bg_node_list = [sum(bg_node_list[:i+1]) for i in
                             range(len(bg_node_list))]

        for (src, dst) in super_g.edges():
            a_src = random.randint((0 if src == 0 else
                                    accu_bg_node_list[src-1]),
                                   accu_bg_node_list[src]-1)
            a_dst = random.randint((0 if dst == 0 else
                                    accu_bg_node_list[dst-1]),
                                   accu_bg_node_list[dst]-1)
            g.add_edges([a_src], [a_dst])
            # add super edge type?

        g = self.to_networkx(g)

        return g

    def gen_connect_hyper_with_config(self, g_class, graph_num_per_class):
        super_g = nx.Graph()
        super_g.add_nodes_from(range(sum(graph_num_per_class)))
        graph_num_per_class = np.cumsum(graph_num_per_class)
        for src in range(super_g.number_of_nodes()):
            src_class = bisect.bisect_left(graph_num_per_class, src)
            for dst in range(src, super_g.number_of_nodes()):
                if src == dst:
                    continue
                dst_class = bisect.bisect_left(graph_num_per_class, dst)
                prob = self.default_gen_param['link_prob'][src_class] * self.default_gen_param['link_prob'][dst_class]
                if np.random.binomial(1, prob, 1):
                    super_g.add_edge(src, dst)

        return super_g

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

    def gen_component_from_dataset(self, dataset, label=0):
        # dataset statistics inquiry
        # assert required label class is actually there
        g, feat = dataset.sample(label)

        return g, feat

    def gen_label(self, ratio_list, g_class):
        '''
        for now we assume label is the majority vote
        '''
        if self.hyper_gen_param['label_type'] == 'config':
            return g_class

        elif self.hyper_gen_param['label_type'] == 'majority':
            return ratio_list.index(max(ratio_list))








    def gen_component(self, feature_type, feature_params,
                      min_nodes, max_nodes, min_deg, max_deg):
        num_n = int(random.randint(min_nodes, max_nodes)/2)*2
        deg = int(random.randint(min_deg, max_deg)/2)*4
        g = nx.random_regular_graph(deg, num_n)
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
        else:
            raise NotImplementedError

        return g, feat
