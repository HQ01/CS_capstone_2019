import pickle
from dataset.tu import DiffpoolDataset
from dataset.syncdataset import SyncPoolDataset
from dataset.dummydataset import DummyClique, DummyWrapper
from dataset.sync_expand_dataset import SyncExpandDataset
import networkx as nx

if __name__ == '__main__':
    #clique_gen = DummyClique(600, [10,20,30,40,50],[20,30,40,50,60], 5)

    #with open('./EXP.pkl', 'rb') as f:
    #    data_list = pickle.load(f)
    #wrapper_gen = DummyWrapper(data_list)
    d_k_a = {'feature_mode': 'default', 'assign_feat':'id'}
    #d_k_a = {'feature_mode': 'deg-num', 'assign_feat':'id'}
    #print("finished cliquegen")
    diffpool_gen = DiffpoolDataset('ENZYMES', use_node_attr=True,
                                   use_node_label=False,
                                   mode='train',
                                   train_ratio=0.8,
                                   test_ratio=0.1,
                                   **d_k_a)
    #sync_expand_gen = SyncExpandDataset(1, base_graph_gen=diffpool_gen)
    #print("diffpool_gen finished")
    #print("diffpool_gen [0] is", type(diffpool_gen[0][0]))
    dataset = SyncPoolDataset(600, graph_dataset=diffpool_gen, num_sub_graphs=10,
                              mode='train')
    pickle_out = open('sync_fix_H.pickle', 'wb')
    #pickle_out = open('sync_fix_2.pickle', 'wb')
    pickle.dump(dataset, pickle_out)
    #pickle.dump(dataset, pickle_out)
    pickle_out.close()

    diffpool_gen.set_mode('val')
    #sync_expand_gen_val = SyncExpandDataset(3, base_graph_gen=diffpool_gen)
    dataset_val = SyncPoolDataset(120, graph_dataset=diffpool_gen, num_sub_graphs=10, mode='val')
    pickle_out = open('sync_fix_H_val.pickle', 'wb')
    pickle.dump(dataset_val, pickle_out)
    pickle_out.close()

    diffpool_gen.set_mode('test')
    #sync_expand_gen_test = SyncExpandDataset(3, base_graph_gen=diffpool_gen)
    dataset_test = SyncPoolDataset(120, graph_dataset=diffpool_gen, num_sub_graphs=10, mode='test')
    pickle_out = open('sync_fix_H_test.pickle', 'wb')
    pickle.dump(dataset_test, pickle_out)
    pickle_out.close()
