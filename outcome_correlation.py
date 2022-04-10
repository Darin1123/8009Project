import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import shutil


def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return adj, deg_inv_sqrt


def gen_normalized_adjs(adj, D_isqrt):
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def get_labels_from_name(labels, split_idx):
    if isinstance(labels, list):
        labels = list(labels)
        if len(labels) == 0:
            return torch.tensor([])
        for idx, i in enumerate(list(labels)):
            labels[idx] = split_idx[i]
        residual_idx = torch.cat(labels)
    else:
        residual_idx = split_idx[labels]
    return residual_idx


def general_outcome_correlation(adj, y, alpha, num_propagations, post_step, alpha_term, device='cpu', display=True):
    """general outcome correlation. alpha_term = True for outcome correlation, alpha_term = False for residual correlation"""
    adj = adj.to(device)
    orig_device = y.device
    y = y.to(device)
    result = y.clone()
    for _ in tqdm(range(num_propagations), disable=not display):
        result = alpha * (adj @ result)
        if alpha_term:
            result += (1 - alpha) * y
        else:
            result += y
        result = post_step(result)
    return result.to(orig_device)


def label_propagation(data, split_idx, A, alpha, num_propagations, idxs):
    labels = data.y.data
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    label_idx = get_labels_from_name(idxs, split_idx)
    y[label_idx] = F.one_hot(labels[label_idx], c).float().squeeze(1)
    return general_outcome_correlation(A, y, alpha, num_propagations, post_step=lambda x: torch.clamp(x, 0, 1),
                                       alpha_term=True)


def prepare_folder(name, model):
    model_dir = f'models/{name}'

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in model.parameters())}\n')
    return model_dir
