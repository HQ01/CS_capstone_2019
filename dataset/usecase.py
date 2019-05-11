import torch
import dgl

from dataset.syncdataset import SyncPoolDataset

test_sync_pool = SyncPoolDataset(20, num_sub_graphs=10, mode='val')
(candidate_graph, feature), label, n2sub_label = test_sync_pool[1]