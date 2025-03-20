import os
import argparse
import random
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import HGBDataset
from torch_geometric.nn import HANConv
from torch import nn
from typing import Dict, Union
import time
import json

def parse_args():
    parser = argparse.ArgumentParser(description="NC/LP Model Hyperparameters")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels.")
    parser.add_argument("--embedding_size", type=int, default=32, help="Size of the embedding layer.")
    parser.add_argument("--num_epochs", type=int, default=250, help="Number of training epochs.")
    parser.add_argument("--lr_NC", type=float, default=0.01, help="Learning rate for Node Classification.")
    parser.add_argument("--lr_LP", type=float, default=0.01, help="Learning rate for Link Prediction.")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds to run (starting from 1).")
    parser.add_argument("--save_results", type=bool, default=False, help="Set to True if results and config should be saved in runs dir)")
    parser.add_argument("--two_layers", type=bool, default=False, help="Set to True if you want 2 layers instead of 3 in the models")
    parser.add_argument("--msg_split_ratio", type=float, default=0.3, help="Fraction of message passing edges in training for LP (the rest become supervision edges)")
    parser.add_argument("--print_epochs", type=bool, default=False, help="Should epoch scores be printed (True) or not (False)?")
    return parser.parse_known_args()[0]

def setup_determinism(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class HAN_NC(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        embedding_size: int,
        hidden_channels=128,
        heads=8,
        metadata=None
    ):
        super().__init__()
        self.han_conv = HANConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=0.6,
            metadata=metadata
        )
        self.lin = nn.Linear(hidden_channels, embedding_size)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['paper'])
        return out

def train_node_classification(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[data['paper'].train_mask], data['paper'].y[data['paper'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_node_classification(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out.argmax(dim=-1)
        correct = (pred[mask] == data['paper'].y[mask].to(pred.device)).sum()
        acc = correct.float() / mask.sum()
    return acc.item()


def split_train_edges(data, split_ratio):
    total_edges = data['paper', 'train_is', 'label'].edge_index.size(1)
    num_msg_edges = int(total_edges * split_ratio)
    perm = torch.randperm(total_edges)
    msg_edges = perm[:num_msg_edges]
    sup_edges = perm[num_msg_edges:]
    msg_edge_index = data['paper', 'train_is', 'label'].edge_index[:, msg_edges]
    sup_edge_index = data['paper', 'train_is', 'label'].edge_index[:, sup_edges]
    return msg_edge_index, sup_edge_index

def my_negative_sampling(pos_edge_index, num_paper_nodes, num_classes, device):
    neg_edge_index = torch.zeros((2, num_paper_nodes * (num_classes - 1)), device=device)
    neg_edge_index[0, :] = pos_edge_index[0].repeat(num_classes - 1)
    labels = torch.arange(num_classes, device=device)
    
    for i in range(num_paper_nodes):
        positive_label = pos_edge_index[1, i].item()
        remaining_labels = labels[labels != positive_label]
        neg_edge_index[1, i * (num_classes - 1):(i + 1) * (num_classes - 1)] = remaining_labels
    
    return neg_edge_index.to(torch.int32)

def prepare_data(data, device, num_classes, msg_split_ratio):

    unique_classes = data['paper'].y.unique()

    data['label'].x = torch.eye(len(unique_classes), dtype=torch.float32, device=device)

    train_mask_indices = torch.nonzero(data['paper'].train_mask, as_tuple=False).flatten()
    train_label_edges = torch.zeros((2, train_mask_indices.size(0)), dtype=torch.int64, device=device)
    train_label_edges[0] = train_mask_indices
    train_label_edges[1] = data['paper'].y[train_mask_indices]
    data['paper', 'train_is', 'label'].edge_index = train_label_edges

    val_mask_indices = torch.nonzero(data['paper'].val_mask, as_tuple=False).flatten()
    val_label_edges = torch.zeros((2, val_mask_indices.size(0)), dtype=torch.int64, device=device)
    val_label_edges[0] = val_mask_indices
    val_label_edges[1] = data['paper'].y[val_mask_indices]
    data['paper', 'val_is', 'label'].edge_index = val_label_edges

    test_mask_indices = torch.nonzero(data['paper'].test_mask, as_tuple=False).flatten()
    test_label_edges = torch.zeros((2, test_mask_indices.size(0)), dtype=torch.int64, device=device)
    test_label_edges[0] = test_mask_indices
    test_label_edges[1] = data['paper'].y[test_mask_indices]
    data['paper', 'test_is', 'label'].edge_index = test_label_edges


    msg_edge_index, sup_edge_index = split_train_edges(data, msg_split_ratio)

    data['paper', 'train_is', 'label'].edge_index = msg_edge_index
    data['paper', 'train_is', 'label'].edge_label_index = sup_edge_index
    data['paper', 'train_is', 'label'].edge_label = torch.tensor(np.ones((sup_edge_index.size(1))))

    data['paper', 'val_is', 'label'].edge_label_index = data['paper', 'val_is', 'label'].edge_index
    data['paper', 'val_is', 'label'].edge_index = torch.cat((msg_edge_index, sup_edge_index), 1)
    data['paper', 'val_is', 'label'].edge_label = torch.tensor(np.ones((data['paper', 'val_is', 'label'].edge_label_index.size(1))))

    data['paper', 'test_is', 'label'].edge_label_index = data['paper', 'test_is', 'label'].edge_index
    data['paper', 'test_is', 'label'].edge_index = torch.cat(
        (msg_edge_index, sup_edge_index, data['paper', 'val_is', 'label'].edge_label_index), 1
    )
    data['paper', 'test_is', 'label'].edge_label = torch.tensor(
        np.ones((data['paper', 'test_is', 'label'].edge_label_index.size(1)))
    )

    # remember: negative sampling can be here because we use ALL possible negative edges everytime, so the sampling will be the same for each seed
    # if we run on a big dataset where we cannot use all possible negative edges, we need to sample in the training method instead, once every epoch
    data['paper', 'train_is', 'label'].neg_edge_index = my_negative_sampling(data['paper', 'train_is', 'label'].edge_label_index, data['paper', 'train_is', 'label'].edge_label_index.size(1), num_classes, device)

    data = T.ToUndirected()(data)

    return data


def sample_edges(edge_index, num_samples):
    num_edges = edge_index.size(1)
    sampled_indices = torch.randperm(num_edges)[:num_samples]
    sampled_edge_index = edge_index[:, sampled_indices]
    return sampled_edge_index

def create_metapaths_nc(data):
    metapaths = [
        [('paper', 'to', 'author'), ('author', 'to', 'paper')],#PAP
        [('paper','to','subject'), ('subject','to','paper')],#PSP
        [('paper', 'cite', 'paper'),
        ('paper', 'to', 'author'),
        ('author', 'to', 'paper')],#PcPAP
        [('paper', 'cite', 'paper'),
        ('paper', 'to', 'subject'),
        ('subject', 'to', 'paper')],#PcPSP
        [('paper', 'ref', 'paper'),
        ('paper', 'to', 'author'),
        ('author', 'to', 'paper')],#PrPAP
        [('paper', 'cite', 'paper'),
        ('paper', 'to', 'subject'),
        ('subject', 'to', 'paper')],#PrPSP
        ]


    transform = T.AddMetaPaths(
        metapaths=metapaths,
        drop_orig_edge_types=True,
        drop_unconnected_node_types=True
    )

    nc_data = transform(data.clone())

    nc_data['paper', 'metapath_1', 'paper'].edge_index = sample_edges(
        nc_data['paper', 'metapath_1', 'paper'].edge_index, 300000)
    nc_data['paper', 'metapath_3', 'paper'].edge_index = sample_edges(
        nc_data['paper', 'metapath_3', 'paper'].edge_index, 300000)
    nc_data['paper', 'metapath_5', 'paper'].edge_index = sample_edges(
        nc_data['paper', 'metapath_5', 'paper'].edge_index, 300000)

    print("NC Data after adding metapaths:")
    print(nc_data)
    return nc_data

def create_metapaths_lp(data):
    metapaths = [
    [('paper', 'to', 'author'), ('author', 'to', 'paper')],#PAP
    [('paper','to','subject'), ('subject','to','paper')],#PSP
    [('paper', 'cite', 'paper'),
     ('paper', 'to', 'author'),
     ('author', 'to', 'paper')],#PcPAP
    [('paper', 'cite', 'paper'),
     ('paper', 'to', 'subject'),
     ('subject', 'to', 'paper')],#PcPSP
    [('paper', 'ref', 'paper'),
     ('paper', 'to', 'author'),
     ('author', 'to', 'paper')],#PrPAP
    [('paper', 'cite', 'paper'),
     ('paper', 'to', 'subject'),
     ('subject', 'to', 'paper')],#PrPSP
     ]

    transform = T.AddMetaPaths(
        metapaths=metapaths,
        drop_orig_edge_types=False,
        drop_unconnected_node_types=True
    )
    lp_data = transform(data.clone())
 
    lp_data['paper', 'metapath_1', 'paper'].edge_index = sample_edges(
        lp_data['paper', 'metapath_1', 'paper'].edge_index, 300000)
    lp_data['paper', 'metapath_3', 'paper'].edge_index = sample_edges(
        lp_data['paper', 'metapath_3', 'paper'].edge_index, 300000)
    lp_data['paper', 'metapath_5', 'paper'].edge_index = sample_edges(
        lp_data['paper', 'metapath_5', 'paper'].edge_index, 300000)

    print("LP Data after adding metapaths:")
    print(lp_data)
    return lp_data

class EdgeDecoder(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.lin = torch.nn.Linear(embedding_size, 1)

    def forward(self, z_paper, z_label):
        x = z_paper * z_label
        out = self.lin(x).squeeze(-1)
        return out
    
class HAN_LP(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        embedding_size: int,
        hidden_channels=128,
        heads=8,
        metadata=None
    ):
        super().__init__()
        self.han_conv = HANConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=0.6,
            metadata=metadata
        )
        self.lin = nn.Linear(hidden_channels, embedding_size)
        self.decoder = EdgeDecoder(embedding_size)

    def forward(self, x_dict, edge_index_dict):
        h = self.han_conv(x_dict, edge_index_dict)
        h['paper'] = self.lin(h['paper'])
        h['label'] = self.lin(h['label'])
        return h
    
    def decode(self, z_dict, edge_label_index):
        z_paper = z_dict['paper']
        z_label = z_dict['label']
        src, dst = edge_label_index
        out = self.decoder(z_paper[src], z_label[dst])
        return out
    
def train_link_prediction(model, data, optimizer, criterion, train_edge_index_dict):
    model.train()
    optimizer.zero_grad()
    
    z_dict = model(data.x_dict, train_edge_index_dict)
    pos_edge_index = data['paper', 'train_is', 'label'].edge_label_index
    pos_pred = model.decode(z_dict, pos_edge_index)
    neg_edge_index = data['paper', 'train_is', 'label'].neg_edge_index
    neg_pred = model.decode(z_dict, neg_edge_index)

    pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
    neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_link_prediction(model, data, mask, edge_dict):
    model.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, edge_dict)
        paper_embeddings = z_dict['paper'][mask]
        label_embeddings = z_dict['label']
        z_paper = paper_embeddings.unsqueeze(1)
        z_label = label_embeddings.unsqueeze(0)
        x = z_paper * z_label
        x = x.view(-1, x.size(-1))
        pred_scores = model.decoder.lin(x)
        pred_scores = pred_scores.view(z_paper.size(0), z_label.size(1))
        predicted_labels = pred_scores.argmax(dim=1)
        true_labels = data['paper'].y[mask]
        correct = (predicted_labels == true_labels).sum().item()
        accuracy = correct / len(true_labels)
    return accuracy

def save_run(test_acc_NC, test_acc_LP, hidden_channels, embedding_size, num_epochs, lr_NC, lr_LP, num_seeds, msg_split_ratio):
    run_name = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("runs", "HAN_ACM", run_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    np.save(os.path.join(save_path, 'test_acc_NC'), test_acc_NC)
    np.save(os.path.join(save_path, 'test_acc_LP'), test_acc_LP)

    config = {'hidden_channels': hidden_channels, 'embedding_size': embedding_size, 'num_epochs': num_epochs, 'lr_NC': lr_NC, 'lr_LP': lr_LP, 'num_seeds': num_seeds, 'msg_split_ratio': msg_split_ratio}
    with open(os.path.join(save_path, "config.json"), "w") as outfile: 
        json.dump(config, outfile)

def main():
    args = parse_args()
    hidden_channels = args.hidden_channels
    embedding_size = args.embedding_size
    num_epochs = args.num_epochs
    lr_NC = args.lr_NC
    lr_LP = args.lr_LP
    num_seeds = args.num_seeds
    save_results = args.save_results
    msg_split_ratio = args.msg_split_ratio
    print_epochs = args.print_epochs

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    seed_list = range(1, num_seeds + 1)
    dataset = HGBDataset(root='data/ACM', name='ACM')
    og_data = dataset[0]
    test_acc_NC = []
    test_acc_LP = []
    
    for seed in seed_list:
        print(f"Running for seed {seed}")
        
        setup_determinism(seed)
    
        transform = RandomNodeSplit(num_val=0.1, num_test=0.1)
        data = transform(og_data.clone()).to(device)
        num_classes = data['paper'].y.max().item() + 1

        nc_data = create_metapaths_nc(data)

        in_channels_dict_NC = {
            'paper': nc_data['paper'].num_features,
        }
        model_NC = HAN_NC(in_channels_dict_NC, embedding_size, metadata=nc_data.metadata()).to(device)

        with torch.no_grad():
            out = model_NC(nc_data.x_dict, nc_data.edge_index_dict)

        optimizer_NC = torch.optim.Adam(model_NC.parameters(), lr=lr_NC)
        criterion_NC = torch.nn.CrossEntropyLoss()
    
        start_patience = 100
        best_val_nc = 0
        best_test_nc = 0

        for epoch in range(1, num_epochs + 1):
            loss_nc = train_node_classification(model_NC, nc_data, optimizer_NC, criterion_NC)
            train_nc = evaluate_node_classification(model_NC, nc_data, nc_data['paper'].train_mask)
            val_nc = evaluate_node_classification(model_NC, nc_data, nc_data['paper'].val_mask)
            test_nc = evaluate_node_classification(model_NC, nc_data, nc_data['paper'].test_mask)
            if print_epochs:
                print(f'Epoch: {epoch:03d}, '
                      f'Loss: {loss_nc:.4f}, '
                      f'Train Acc: {train_nc:.4f}, '
                      f'Val Acc: {val_nc:.4f}, '
                      f'Test Acc: {test_nc:.4f}')
            if val_nc > best_val_nc:
                best_val_nc = val_nc
                best_test_nc = test_nc
                patience = start_patience
            else:
                patience -= 1

            if patience <= 0:
                print('Early stopping due to no improvement in validation accuracy.')
                break

        print(f"NC Test Accuracy for seed {seed}: {best_test_nc:.4f}")
        test_acc_NC.append(best_test_nc)

        lp_data = prepare_data(data, device, num_classes, msg_split_ratio)    
        lp_data = create_metapaths_lp(lp_data)

        train_edge_index_dict = {
          ('paper','PAP','paper'): lp_data['paper','metapath_0','paper'].edge_index,
          ('paper','PSP','paper'): lp_data['paper','metapath_1','paper'].edge_index,
          ('paper','PcPAP','paper'): lp_data['paper','metapath_2','paper'].edge_index,
          ('paper','PcPSP','paper'): lp_data['paper','metapath_3','paper'].edge_index,
          ('paper','PrPAP','paper'): lp_data['paper','metapath_4','paper'].edge_index,
          ('paper','PrPSP','paper'): lp_data['paper','metapath_5','paper'].edge_index,
          ('paper','is','label'): lp_data['paper','train_is','label'].edge_index,
          ('label','rev_is','paper'): lp_data['label','rev_train_is','paper'].edge_index,
        }

        val_edge_index_dict = {
          ('paper','PAP','paper'): lp_data['paper','metapath_0','paper'].edge_index,
          ('paper','PSP','paper'): lp_data['paper','metapath_1','paper'].edge_index,
          ('paper','PcPAP','paper'): lp_data['paper','metapath_2','paper'].edge_index,
          ('paper','PcPSP','paper'): lp_data['paper','metapath_3','paper'].edge_index,
          ('paper','PrPAP','paper'): lp_data['paper','metapath_4','paper'].edge_index,
          ('paper','PrPSP','paper'): lp_data['paper','metapath_5','paper'].edge_index,
          ('paper','is','label'): lp_data['paper','val_is','label'].edge_index,
          ('label','rev_is','paper'): lp_data['label','rev_val_is','paper'].edge_index,
        }

        test_edge_index_dict = {
          ('paper','PAP','paper'): lp_data['paper','metapath_0','paper'].edge_index,
          ('paper','PSP','paper'): lp_data['paper','metapath_1','paper'].edge_index,
          ('paper','PcPAP','paper'): lp_data['paper','metapath_2','paper'].edge_index,
          ('paper','PcPSP','paper'): lp_data['paper','metapath_3','paper'].edge_index,
          ('paper','PrPAP','paper'): lp_data['paper','metapath_4','paper'].edge_index,
          ('paper','PrPSP','paper'): lp_data['paper','metapath_5','paper'].edge_index,
          ('paper','is','label'): lp_data['paper','test_is','label'].edge_index,
          ('label','rev_is','paper'): lp_data['label','rev_test_is','paper'].edge_index,
        }
        # TO DO!!!! Try with metapaths involving label nodes as well. They will need to be created for each split since label nodes differ between splits

        in_channels_dict_LP = {
            'paper': lp_data['paper'].num_features,
            'author': lp_data['author'].num_features,
            'term': lp_data['term'].num_features,
            'subject': lp_data['subject'].num_features,
            'label': lp_data['label'].num_features,

        }
        metadata = (
            lp_data.node_types, 
            [
               ('paper','PAP','paper'),
               ('paper','PSP','paper'),
               ('paper','PcPAP','paper'),
               ('paper','PcPSP','paper'),
               ('paper','PrPAP','paper'),
               ('paper','PrPSP','paper'),
               ('paper','is','label'),
               ('label','rev_is','paper'),
            ]
        )
        model_LP = HAN_LP(in_channels_dict_LP, embedding_size, metadata=metadata).to(device)
        optimizer_LP = torch.optim.Adam(model_LP.parameters(), lr=lr_LP)
        criterion_LP = torch.nn.BCEWithLogitsLoss()

        best_val_lp = 0
        best_test_lp = 0

        for epoch in range(1, num_epochs + 1):
            loss_lp = train_link_prediction(model_LP, lp_data, optimizer_LP, criterion_LP, train_edge_index_dict)
            train_lp = evaluate_link_prediction(model_LP, lp_data, lp_data['paper'].train_mask, train_edge_index_dict)
            val_lp = evaluate_link_prediction(model_LP, lp_data, lp_data['paper'].val_mask, val_edge_index_dict)
            test_lp = evaluate_link_prediction(model_LP, lp_data, lp_data['paper'].test_mask, test_edge_index_dict)
            if print_epochs:
                print(f'Epoch: {epoch:03d}, '
                      f'Loss: {loss_lp:.4f}, '
                      f'Train Acc: {train_lp:.4f}, '
                      f'Val Acc: {val_lp:.4f}, '
                      f'Test Acc: {test_lp:.4f}')
            if val_lp > best_val_lp:
                best_val_lp = val_lp
                best_test_lp = test_lp
                patience = start_patience
            else:
                patience -= 1

            if patience <= 0:
                print('Early stopping due to no improvement in validation accuracy.')
                break

        test_acc_LP.append(best_test_lp)
        print(f"LP Test Accuracy for seed {seed}: {best_test_lp:.4f}\n")
        
    if save_results:
        save_run(test_acc_NC, test_acc_LP, hidden_channels, embedding_size, num_epochs, lr_NC, lr_LP, num_seeds, msg_split_ratio)

if __name__=="__main__":
    main()