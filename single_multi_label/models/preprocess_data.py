import numpy as np
import torch
import scipy
from torch_geometric.data import Data
import os
import pathlib
# TAKEN FROM  "Multi-label Node Classification On Graph-Structured Data"

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_blog():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    blog_path = os.path.abspath(os.path.join(script_dir, "../data/blog"))
    mat_path = os.path.join(blog_path, "blogcatalog.mat")
    print('Loading dataset ' + mat_path)
    mat = scipy.io.loadmat(mat_path)
    labels = mat['group']
    labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).long()
    edge_index = torch.transpose(torch.nonzero(adj.to_dense()), 0, 1).long()
    # prepare the feature matrix
    features = torch.eye(labels.shape[0])

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    y = labels.clone().detach().float()

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)

    return G


def load_dblp(path="../data/dblp/"):
    print("Absolute path for features.txt:", os.path.abspath(os.path.join(path, "features.txt")))
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dblp_path = os.path.abspath(os.path.join(script_dir, "../data/dblp"))
    labels_path = os.path.join(dblp_path, "labels.txt")
    labels = np.genfromtxt(labels_path, dtype=np.float32, delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.FloatTensor(np.genfromtxt(os.path.join(dblp_path, "features.txt"),
                                               delimiter=",", dtype=np.float64))
    edge_list = torch.tensor(np.genfromtxt(os.path.join(dblp_path, "dblp.edgelist"))).long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (labels.shape[0], labels.shape[0]))


    G = Data(x=features,
             edge_index=edge_index,
             y=labels)

    return G
