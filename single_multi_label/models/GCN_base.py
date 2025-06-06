import os
import argparse
import random
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.nn import GCNConv, to_hetero
from torch_geometric.utils import negative_sampling
import time
import json
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GraphConv, HeteroConv
from sklearn.metrics import average_precision_score
from preprocess_data import load_blog, load_dblp
from torch_geometric.datasets import Yelp



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
    parser.add_argument("--ds", type=str, default='DBLP', help="Which dataset to run on.")
    parser.add_argument("--neg_pos_ratio", type=int, default=-1, help="How many times the amount of positive edges should the amount of negative edges be? -1 if all negative edges should be used")
    parser.add_argument("--print_epochs", type=bool, default=False, help="Should epoch scores be printed (True) or not (False)?")


    
    return parser.parse_known_args()[0]

    
def ap_score(y_true, y_pred):
    return average_precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro")

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

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, two_layers, add_self_loops=True):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels, add_self_loops=add_self_loops, normalize=True)
        self.conv2 = GCNConv(hidden_channels, 48, add_self_loops=add_self_loops, normalize=True)
        self.conv3 = GCNConv(48, embedding_size, add_self_loops=add_self_loops, normalize=True)

        self.conv1b = GCNConv(-1, hidden_channels, add_self_loops=add_self_loops, normalize=True)
        self.conv2b = GCNConv(hidden_channels, embedding_size, add_self_loops=add_self_loops, normalize=True)

        self.two_layers = two_layers

    def forward(self, x, edge_index):
        if self.two_layers:
            x = self.conv1b(x, edge_index)
            x = torch.relu(x)
            x = self.conv2b(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            x = torch.relu(x)
            x = self.conv3(x, edge_index)
        return x

class GCN_NC(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_classes, two_layers):
        super().__init__()
        self.encoder = GCN(hidden_channels, embedding_size, two_layers)
        self.decoder = torch.nn.Linear(embedding_size, num_classes)
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z)
        return out


def train_node_classification(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask]).float()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_node_classification(model, data, mask, og_data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.sigmoid(out[mask])
        true_labels = data.y[mask].float()
        precision = ap_score(true_labels, probs)

    return precision.item()


def sub_data(data, mask, device):
    s_data = HeteroData()
    s_data['paper'].x = data.x
    s_data['paper', 'cites', 'paper'].edge_index = data.edge_index
    num_labels = data.y.size(1)
    s_data['label'].x = torch.eye(num_labels, dtype=torch.float32, device=device)
    mask_indices = torch.nonzero(mask, as_tuple=False).flatten()#nodes that will be wired
    mask_y = data.y[mask_indices]
    num_edges_to_create = int(mask_y.sum().item())
    label_edges = torch.zeros((2, num_edges_to_create), dtype=torch.int64, device=device)
    curr = 0 #array pointer
    for i in range(len(mask_indices)):
        for j in range(num_labels):
            if mask_y[i, j] == 1:
                label_edges[0, curr] = mask_indices[i]
                label_edges[1, curr] = j
                curr += 1
    s_data['paper', 'is', 'label'].edge_index = label_edges
    return s_data

def split_train_edges(data, split_ratio):
    total_edges = data['paper', 'is', 'label'].edge_index.size(1)
    num_msg_edges = int(total_edges * split_ratio)
    perm = torch.randperm(total_edges)
    msg_edges = perm[:num_msg_edges]
    sup_edges = perm[num_msg_edges:]
    msg_edge_index = data['paper', 'is', 'label'].edge_index[:, msg_edges]
    sup_edge_index = data['paper', 'is', 'label'].edge_index[:, sup_edges]
    return msg_edge_index, sup_edge_index

def my_negative_sampling(pos_edge_index, num_paper_nodes, num_classes, device):
    neg_edge_index = torch.zeros((2, num_paper_nodes * (num_classes - 1)), device=device)
    neg_edge_index[0, :] = pos_edge_index[0].repeat_interleave(num_classes - 1)
    labels = torch.arange(num_classes, device=device)
    
    for i in range(num_paper_nodes):
        positive_label = pos_edge_index[1, i].item()
        remaining_labels = labels[labels != positive_label]
        neg_edge_index[1, i * (num_classes - 1):(i + 1) * (num_classes - 1)] = remaining_labels
    return neg_edge_index.to(torch.int32)

def prepare_lp_data(data, train_mask, val_mask, test_mask, device, num_classes, msg_split_ratio, neg_pos_ratio):
    train_data_LP = sub_data(data, train_mask, device)
    val_data_LP = sub_data(data, val_mask, device)
    test_data_LP = sub_data(data, test_mask, device)
    msg_edge_index, sup_edge_index = split_train_edges(train_data_LP, msg_split_ratio)

    train_data_LP['paper', 'is', 'label'].edge_index = msg_edge_index
    train_data_LP['paper', 'is', 'label'].edge_label_index = sup_edge_index
    train_data_LP['paper', 'is', 'label'].edge_label = torch.tensor(np.ones((sup_edge_index.size(1))))

    val_data_LP['paper', 'is', 'label'].edge_label_index = val_data_LP['paper', 'is', 'label'].edge_index
    val_data_LP['paper', 'is', 'label'].edge_index = torch.cat((msg_edge_index, sup_edge_index), 1)
    val_data_LP['paper', 'is', 'label'].edge_label = torch.tensor(np.ones((val_data_LP['paper', 'is', 'label'].edge_label_index.size(1))))

    test_data_LP['paper', 'is', 'label'].edge_label_index = test_data_LP['paper', 'is', 'label'].edge_index
    test_data_LP['paper', 'is', 'label'].edge_index = torch.cat(
        (msg_edge_index, sup_edge_index, val_data_LP['paper', 'is', 'label'].edge_label_index), 1
    )
    test_data_LP['paper', 'is', 'label'].edge_label = torch.tensor(
        np.ones((test_data_LP['paper', 'is', 'label'].edge_label_index.size(1)))
    )
    if neg_pos_ratio == -1:
        # remember: negative sampling can be here because we use ALL possible negative edges everytime, so the sampling will be the same for each seed
        # if we run on a big dataset where we cannot use all possible negative edges, we need to sample in the training method instead, once every epoch
        train_data_LP['paper', 'is', 'label'].neg_edge_index = my_negative_sampling(train_data_LP['paper', 'is', 'label'].edge_label_index, train_data_LP['paper', 'is', 'label'].edge_label_index.size(1), num_classes, device)

    train_data_LP = T.ToUndirected()(train_data_LP)
    val_data_LP = T.ToUndirected()(val_data_LP)
    test_data_LP = T.ToUndirected()(test_data_LP)

    return train_data_LP, val_data_LP, test_data_LP

class HeteroGCN(torch.nn.Module):
    # important: GCN on homogeneous paper-paper edges and GraphConv on cross-type edges
    
    def __init__(self, hidden_channels, embed_dim, metadata, two_layers):
        super().__init__()
        self.two_layers = two_layers

        def make_layer(in_dim, out_dim):
            return HeteroConv({
                ('paper', 'cites', 'paper'): GCNConv(in_dim, out_dim),
                ('paper', 'is',   'label'):  GraphConv(in_dim, out_dim),
                ('label', 'rev_is', 'paper'): GraphConv(in_dim, out_dim),
            }, aggr='mean')

        self.layer1 = make_layer(-1, hidden_channels)
        if two_layers:
            self.layer2 = make_layer(hidden_channels, embed_dim)
        else:
            self.layer2 = make_layer(hidden_channels, 48)
            self.layer3 = make_layer(48, embed_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: torch.relu(v) for k, v in self.layer1(x_dict, edge_index_dict).items()}
        if self.two_layers:
            return self.layer2(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in self.layer2(x_dict, edge_index_dict).items()}
        return self.layer3(x_dict, edge_index_dict)

class EdgeDecoder(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.lin = torch.nn.Linear(embedding_size, 1)

    def forward(self, z_paper, z_label):
        x = z_paper * z_label
        out = self.lin(x).squeeze(-1)
        return out

class LinkPredictor(torch.nn.Module):
    def __init__(self, hidden, emb, metadata, two_layers):
        super().__init__()
        self.encoder = HeteroGCN(hidden, emb, metadata, two_layers)
        self.decoder = EdgeDecoder(emb)

    def forward(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict, edge_label_index):
        z_paper, z_label = z_dict['paper'], z_dict['label']
        src, dst = edge_label_index
        return self.decoder(z_paper[src], z_label[dst])


def train_link_prediction(model, data, optimizer, criterion, num_classes, device, neg_pos_ratio):
    model.train()
    optimizer.zero_grad()
    z_dict = model(data.x_dict, data.edge_index_dict)
    pos_edge_index = data['paper', 'is', 'label'].edge_label_index
    pos_pred = model.decode(z_dict, pos_edge_index)
    neg_edge_index = None
    if neg_pos_ratio > -1:
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(data['paper'].num_nodes, data['label'].num_nodes),
            num_neg_samples=neg_pos_ratio*pos_edge_index.size(1),
            method='sparse'
        )
    else:
        neg_edge_index = data['paper', 'is', 'label'].neg_edge_index

    neg_pred = model.decode(z_dict, neg_edge_index)
    pos_loss = criterion(pos_pred, torch.ones_like(pos_pred,  dtype=torch.float))
    neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred, dtype=torch.float))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_link_prediction(model, data, mask, og_data):
    model.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
        paper_embeddings = z_dict['paper'][mask]
        label_embeddings = z_dict['label']
        z_paper = paper_embeddings.unsqueeze(1)
        z_label = label_embeddings.unsqueeze(0)
        x = z_paper * z_label
        x = x.view(-1, x.size(-1))
        pred_scores = model.decoder.lin(x)
        pred_scores = pred_scores.view(z_paper.size(0), z_label.size(1))
        pred_probs = torch.sigmoid(pred_scores)
        true_labels = og_data.y[mask].float()
        precision = ap_score(true_labels, pred_probs)

    return precision.item()

    

def save_run(test_ap_NC, test_ap_LP, hidden_channels, embedding_size, num_epochs, lr_NC, lr_LP, num_seeds, two_layers, msg_split_ratio, ds, neg_pos_ratio):
    run_name = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("runs", "GCN_"+ds, run_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    np.save(os.path.join(save_path, 'test_ap_NC'), test_ap_NC)
    np.save(os.path.join(save_path, 'test_ap_LP'), test_ap_LP)

    config = {'hidden_channels': hidden_channels, 'embedding_size': embedding_size, 'num_epochs': num_epochs, 'lr_NC': lr_NC, 'lr_LP': lr_LP, 'num_seeds': num_seeds, 'two_layers': two_layers, 'msg_split_ratio': msg_split_ratio, 'dataset': ds, 'neg_pos_ratio': neg_pos_ratio}
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
    two_layers = args.two_layers
    msg_split_ratio = args.msg_split_ratio
    ds = args.ds
    neg_pos_ratio = args.neg_pos_ratio
    print_epochs = args.print_epochs


    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    seed_list = range(1, num_seeds + 1)
    dataset = None
    if ds == 'blog':
        dataset=load_blog()
        seed_list = range(1,4)
    elif ds == 'DBLP':
        dataset=load_dblp()
    elif ds =='Yelp':
        dataset=Yelp(root='data/yelp')
        dataset = dataset[0]

    else:
        print("Invalid dataset name.")
        return
    
    
    print("Running GCN on dataset:", ds)

    og_data = dataset
    og_data = T.ToUndirected()(og_data).to(device)
    
    test_ap_NC = []
    test_ap_LP = []
    
    for seed in seed_list:
        print(f"Running for seed {seed}")
        
        setup_determinism(seed)
    
        transform = RandomNodeSplit(num_val=0.1, num_test=0.1)
        data = transform(og_data.clone()).to(device)

        if ds == 'blog':
            if seed == 1:
                split_dict = torch.load("data/blog/split_0.pt")
            elif seed == 2:
                split_dict = torch.load("data/blog/split_1.pt")
            elif seed == 3:
                split_dict = torch.load("data/blog/split_2.pt")

            data.train_mask = split_dict['train_mask']
            data.val_mask = split_dict['val_mask']
            data.test_mask = split_dict['test_mask']
        else:
            transform = RandomNodeSplit(num_val=0.1, num_test=0.1)
            data = transform(data)

        num_classes = data.y.size(1) 
    
        model_NC = GCN_NC(hidden_channels, embedding_size, num_classes, two_layers).to(device)
        optimizer_NC = torch.optim.Adam(model_NC.parameters(), lr=lr_NC)
        criterion_NC = torch.nn.BCEWithLogitsLoss()
    
        start_patience = 100
        best_val_nc = 0
        best_test_nc = 0

        for epoch in range(1, num_epochs + 1):
            loss_nc = train_node_classification(model_NC, data, optimizer_NC, criterion_NC)
            train_nc = evaluate_node_classification(model_NC, data, data.train_mask, og_data)
            val_nc = evaluate_node_classification(model_NC, data, data.val_mask, og_data)
            test_nc = evaluate_node_classification(model_NC, data, data.test_mask, og_data)
            if print_epochs:
                print(f'Epoch: {epoch:03d}, '
                      f'Loss: {loss_nc:.4f}, '
                      f'Train ap: {train_nc:.4f}, '
                      f'Val ap: {val_nc:.4f}, '
                      f'Test ap: {test_nc:.4f}')
            if val_nc > best_val_nc:
                best_val_nc = val_nc
                best_test_nc = test_nc
                patience = start_patience
            else:
                patience -= 1

            if patience <= 0:
                print('Early stopping due to no improvement in validation AP.')
                break

        print(f"NC Test AP for seed {seed}: {best_test_nc:.4f}")
        test_ap_NC.append(best_test_nc)

        train_data_LP, val_data_LP, test_data_LP = prepare_lp_data(data, data.train_mask, data.val_mask, data.test_mask, device, num_classes, msg_split_ratio, neg_pos_ratio)
    
        train_data_LP = train_data_LP.to(device)
        val_data_LP = val_data_LP.to(device)
        test_data_LP = test_data_LP.to(device)

        for d in (train_data_LP, val_data_LP, test_data_LP):
            ei = d['paper', 'cites', 'paper'].edge_index
            ei, _ = add_self_loops(
                ei,
                num_nodes=d['paper'].num_nodes
            )
            d['paper', 'cites', 'paper'].edge_index = ei
    
        metadata = train_data_LP.metadata()
        model_LP = LinkPredictor(hidden_channels, embedding_size, metadata, two_layers).to(device)
        optimizer_LP = torch.optim.Adam(model_LP.parameters(), lr=lr_LP)
        criterion_LP = torch.nn.BCEWithLogitsLoss()

        best_val_lp = 0
        best_test_lp = 0

        for epoch in range(1, num_epochs + 1):
            loss_lp = train_link_prediction(model_LP, train_data_LP, optimizer_LP, criterion_LP, num_classes, device, neg_pos_ratio)
            train_lp = evaluate_link_prediction(model_LP, train_data_LP, data.train_mask, og_data)
            val_lp = evaluate_link_prediction(model_LP, val_data_LP, data.val_mask, og_data)
            test_lp = evaluate_link_prediction(model_LP, test_data_LP, data.test_mask, og_data)
            if print_epochs:
                print(f'Epoch: {epoch:03d}, '
                      f'Loss: {loss_lp:.4f}, '
                      f'Train ap: {train_lp:.4f}, '
                      f'Val ap: {val_lp:.4f}, '
                      f'Test ap: {test_lp:.4f}')
            if val_lp > best_val_lp:
                best_val_lp = val_lp
                best_test_lp = test_lp
                patience = start_patience
            else:
                patience -= 1

            if patience <= 0:
                print('Early stopping due to no improvement in validation AP.')
                break

        test_ap_LP.append(best_test_lp)
        print(f"LP Test AP for seed {seed}: {best_test_lp:.4f}\n")
        
    if save_results:
        save_run(test_ap_NC, test_ap_LP, hidden_channels, embedding_size, num_epochs, lr_NC, lr_LP, num_seeds, two_layers, msg_split_ratio, ds, neg_pos_ratio)

if __name__=="__main__":
    main()