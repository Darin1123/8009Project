from tqdm import tqdm
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from scipy import sparse
import numpy as np


np.random.seed(0)


def sgc(x, adj, num_propagations):
    for _ in tqdm(range(num_propagations)):
        x = adj @ x
    return torch.from_numpy(x).to(torch.float)

def diffusion(x, adj, num_propagations, p, alpha):
    if p is None:
        p = 1.
    if alpha is None:
        alpha = 0.5

    x = x **p
    for i in tqdm(range(num_propagations)):
        x = x - alpha * (sparse.eye(adj.shape[0]) - adj) @ x
        x = x **p
    return torch.from_numpy(x).to(torch.float)


def preprocess(data, preprocess="diffusion", num_propagations=10, p=None, alpha=None, use_cache=True, post_fix=""):
    if use_cache:
        try:
            x = torch.load(f'embeddings/{preprocess}{post_fix}.pt')
            print('Using cache')
            return x
        except:
            print(f'embeddings/{preprocess}{post_fix}.pt not found or not enough iterations! Regenerating it now')

    print('Computing adj...')
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj = adj.to_scipy(layout='csr')

    sgc_dict = {}

    print(f'Start {preprocess} processing')

    if preprocess == "sgc":
        result = sgc(data.x.numpy(), adj, num_propagations)
    if preprocess == "diffusion":
        result = diffusion(data.x.numpy(), adj, num_propagations, p = p, alpha = alpha)

    torch.save(result, f'embeddings/{preprocess}{post_fix}.pt')

    return result

