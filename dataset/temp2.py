import pickle

path = './CEP_dataset_output.pkl'

with open(path, 'rb') as f:
    dataset = pickle.load(f)
print(dataset.train[20])

a = dataset.sample(3)
print(a)
