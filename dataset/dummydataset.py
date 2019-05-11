"""
Dummy dataset contains of all clique graphs
"""
from __future__ import absolute_import
import math
import random
import networkx as nx
import numpy as np
from dgl import DGLGraph

class DummyClique():
    """
    Dummy Clique Dataset
    """

    def __init__(self, num_graphs, min_num_v, max_num_v, num_classes):
        super(DummyClique, self).__init__()
        random.seed(0)
        np.random.seed(0)
        assert num_graphs % num_classes == 0, "classes distribution not even!"
        assert len(min_num_v) == len(max_num_v) == num_classes, 'not every\
                class is specified!'
        self.num_graphs = num_graphs
        self.min_num_v = min_num_v
        self.max_num_v = max_num_v
        self.num_classes = num_classes
        self.graphs = []
        self.labels = []
        self.features = []
        self._generate()
        # preprocess
        for i in range(len(self.graphs)):
            print("processing graph number {}".format(i))
            self.graphs[i] = DGLGraph(self.graphs[i])
            nodes = self.graphs[i].nodes()
            self.graphs[i].add_edges(nodes, nodes)

    def __len__(self):
        """
        # graphs in the dataset
        """
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def _generate(self):
        num_graphs = self.num_graphs
        for c in range(self.num_classes):
            for _ in range(num_graphs // self.num_classes):
                self._generate_sub(c, self.min_num_v[c], self.max_num_v[c])
            print("class {} finished generating".format(c))

    def _generate_sub(self, label, min_v, max_v):
        if label == 1:
            #self._generate_star(label, min_v, max_v)
            self._generate_renyi(label, min_v, max_v)
        elif label == 2:
            #self._generate_wheel(label, min_v, max_v)
            self._generate_barabasi(label, min_v, max_v)
        else:
            #self._generate_clique(label, min_v, max_v)
            self._generate_watts(label, min_v, max_v)

    def _generate_clique(self, label, min_v, max_v):
        # now label is tie to # of nodes
        # \TODO move this out sometims
        feature_params = {'mean': 0.0, 'variance': 1.0, 'dim': 32}
        num_v = np.random.randint(self.min_num_v[label], self.max_num_v[label])
        g = nx.complete_graph(num_v)
        feat = np.random.normal(label,
                                feature_params['variance'],
                                (g.number_of_nodes(), feature_params['dim']))
        #feat = np.zeros((g.number_of_nodes(), feature_params['dim']))
        #feat[:,label] = 1.0
        # print(feat)

        self.graphs.append(g)
        self.labels.append(label)
        self.features.append(feat)
        # DGL graph + numpy node feature.

    def _generate_star(self, label, min_v, max_v):
        feature_params = {'mean':0.0, 'variance':0.1, 'dim': 32}
        num_v = np.random.randint(self.min_num_v[label], self.max_num_v[label])
        g = nx.star_graph(num_v - 1)
        feat = np.random.normal(label,
                                feature_params['variance'],
                                (g.number_of_nodes(), feature_params['dim']))
        self.graphs.append(g)
        self.labels.append(label)
        self.features.append(feat)

    def _generate_wheel(self, label, min_v, max_v):
        feature_params = {'mean':0.0, 'variance':0.1, 'dim':32}
        num_v = np.random.randint(self.min_num_v[label], self.max_num_v[label])
        n_rows = np.random.randint(2,num_v//2)
        n_cols = num_v // n_rows
        g = nx.wheel_graph(num_v)
        #g = nx.grid_graph([n_rows, n_cols])
        #g = nx.convert_node_labels_to_integers(g)
        feat = np.random.normal(label,
                                feature_params['variance'],
                                (g.number_of_nodes(), feature_params['dim']))
        self.graphs.append(g)
        self.labels.append(label)
        self.features.append(feat)

    def _generate_renyi(self, label, min_v, max_v):
        p=0.5
        feature_params = {'mean':0.0, 'variance':0.1, 'dim':32}
        num_v = np.random.randint(self.min_num_v[label], self.max_num_v[label])
        g = nx.erdos_renyi_graph(num_v, p)
        feat = np.random.normal(label,
                                feature_params['variance'],
                                (g.number_of_nodes(), feature_params['dim']))
        self.graphs.append(g)
        self.labels.append(label)
        self.features.append(feat)

    def _generate_barabasi(self, label, min_v, max_v):
        ed = 4
        feature_params = {'mean':0.0, 'variance':0.1, 'dim':32}
        num_v = np.random.randint(self.min_num_v[label], self.max_num_v[label])
        g = nx.barabasi_albert_graph(num_v, ed)
        feat = np.random.normal(label,
                                feature_params['variance'],
                                (g.number_of_nodes(), feature_params['dim']))
        self.graphs.append(g)
        self.labels.append(label)
        self.features.append(feat)

    def _generate_watts(self, label, min_v, max_v):
        k = 3
        p = 0.6
        feature_params = {'mean':0.0, 'variance':0.1, 'dim':32}
        num_v = np.random.randint(self.min_num_v[label], self.max_num_v[label])
        g = nx.watts_strogatz_graph(num_v, k, p)
        feat = np.random.normal(label,
                                feature_params['variance'],
                                (g.number_of_nodes(), feature_params['dim']))
        self.graphs.append(g)
        self.labels.append(label)
        self.features.append(feat)






        

    def sample(self, target_label):
        '''
        yield one graph instance of particular label
        '''
        sample_label = -1
        sample_range = len(self.graphs)
        sample_graphs = self.graphs
        sample_labels = self.labels
        sample_features = self.features

        while sample_label != target_label:
            choice = random.randint(0, sample_range-1)
            sample_label = sample_labels[choice]
        sample_graph = sample_graphs[choice]

        return sample_graph, sample_features[choice]

class DummyWrapper():

    def __init__(self, data_list):
        super(DummyWrapper, self).__init__()
        random.seed(0)
        np.random.seed(0)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return (self.data_list[idx][0].adjacency_matrix().to_dense(),\
                self.data_list[idx][0].ndata['feat']), self.data_list[idx][2], None

    def sample(self, target_label):
        sample_label = -1
        sample_range = len(self.data_list)
        while sample_label != target_label:
            choice = random.randint(0, sample_range-1)
            sample_label = self.data_list[choice][2]
        sample_graph = self.data_list[choice][0]

        return sample_graph, sample_graph.ndata['feat']
