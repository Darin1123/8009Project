from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T


class DataLoader:
    def __init__(self, dataset_name='ogbn-arxiv'):
        self.__dataset_name = dataset_name

    def load_data(self):
        dataset = PygNodePropPredDataset(name=self.__dataset_name, root='data', transform=T.ToSparseTensor())
        split_idx = dataset.get_idx_split()
        return dataset, split_idx

    def load_preprocess_data(self):
        return PygNodePropPredDataset(name=self.__dataset_name, root='data')[0]
