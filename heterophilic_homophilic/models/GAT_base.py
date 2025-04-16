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
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.utils import negative_sampling
import time
import json




def parse_args():
    parser = argparse.ArgumentParser(description="NC/LP Model Hyperparameters")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Number of hidden channels.")
    parser.add_argument("--middle_size", type=int, default=48, help="Size of middle layer.")
    parser.add_argument("--embedding_size", type=int, default=32, help="Size of the embedding layer.")
    parser.add_argument("--num_epochs", type=int, default=250, help="Number of training epochs.")
    parser.add_argument("--lr_NC", type=float, default=0.01, help="Learning rate for Node Classification.")
    parser.add_argument("--lr_LP", type=float, default=0.01, help="Learning rate for Link Prediction.")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds to run (starting from 1).")
    parser.add_argument("--save_results", type=bool, default=False, help="Set to True if results and config should be saved in runs dir)")
    parser.add_argument("--two_layers", type=bool, default=False, help="Set to True if you want 2 layers instead of 3 in the models")
    parser.add_argument("--msg_split_ratio", type=float, default=0.3, help="Fraction of message passing edges in training for LP (the rest become supervision edges)")
    parser.add_argument("--ds", type=str, default='CiteSeer', help="Which dataset to run on.")
    parser.add_argument("--neg_pos_ratio", type=int, default=-1, help="How many times the amount of positive edges should the amount of negative edges be? -1 if all negative edges should be used")
    parser.add_argument("--print_epochs", type=bool, default=False, help="Should epoch scores be printed (True) or not (False)?")
    parser.add_argument("--symmetric_sizes", type=bool, default=False, help="If True, the first two layers will be chosen in a triangular shape (False)?")


    
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

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, middle_size, embedding_size, two_layers, heads=8):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), middle_size, heads=heads, concat=True, add_self_loops=False)
        self.conv3 = GATConv((-1, -1), embedding_size, heads=1, concat=False, add_self_loops=False)

        self.conv1b = GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.conv2b = GATConv((-1, -1), embedding_size, heads=1, concat=False, add_self_loops=False)

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

class GAT_NC(torch.nn.Module):
    def __init__(self, hidden_channels, middle_size, embedding_size, num_classes, two_layers, heads=8):
        super().__init__()
        self.encoder = GAT(hidden_channels, middle_size, embedding_size, two_layers, heads=heads)
        self.decoder = torch.nn.Linear(embedding_size, num_classes)
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z)
        return out


def train_node_classification(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_node_classification(model, data, mask, og_data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)
        correct = (pred[mask] == og_data.y[mask].to(pred.device)).sum()
        acc = correct.float() / mask.sum()
    return acc.item()

def get_per_class_acc_nc(model, data, mask, og_data, num_classes, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)

        per_class_acc = torch.zeros(num_classes,dtype=torch.float, device=device)
        for c in range(num_classes):
            correct_in_class = ((pred[mask] == og_data.y[mask].to(pred.device)) & (og_data.y[mask].to(pred.device)==c)).sum()
            total_in_class = (og_data.y[mask] == c).sum()
            per_class_acc[c] = correct_in_class.float()/total_in_class
    return per_class_acc


def sub_data(data, mask, device):
    s_data = HeteroData()
    s_data['paper'].x = data.x
    s_data['paper', 'cites', 'paper'].edge_index = data.edge_index
    unique_classes = data.y.unique()
    s_data['label'].x = torch.eye(len(unique_classes), dtype=torch.float32, device=device)
    mask_indices = torch.nonzero(mask, as_tuple=False).flatten()
    label_edges = torch.zeros((2, mask_indices.size(0)), dtype=torch.int64, device=device)
    label_edges[0] = mask_indices
    label_edges[1] = data.y[mask_indices]
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

class EdgeDecoder(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.lin = torch.nn.Linear(embedding_size, 1)

    def forward(self, z_paper, z_label):
        x = z_paper * z_label
        out = self.lin(x).squeeze(-1)
        return out

class LinkPredictor(torch.nn.Module):
    def __init__(self, hidden_channels, middle_size, embedding_size, metadata, two_layers, heads=8):
        super().__init__()
        self.encoder = GAT(hidden_channels, middle_size, embedding_size, two_layers, heads=heads)
        self.encoder = to_hetero(self.encoder, metadata)
        self.decoder = EdgeDecoder(embedding_size)

    def forward(self, x_dict, edge_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict

    def decode(self, z_dict, edge_label_index):
        z_paper = z_dict['paper']
        z_label = z_dict['label']
        src, dst = edge_label_index
        out = self.decoder(z_paper[src], z_label[dst])
        return out


def train_link_prediction(model, data, optimizer, criterion, num_classes, device, neg_pos_ratio):
    model.train()
    optimizer.zero_grad()
    z_dict = model(data.x_dict, data.edge_index_dict)
    pos_edge_index = data['paper', 'is', 'label'].edge_label_index
    pos_pred = model.decode(z_dict, pos_edge_index)
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
    pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
    neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))
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
        predicted_labels = pred_scores.argmax(dim=1)
        true_labels = og_data.y[mask]
        correct = (predicted_labels == true_labels).sum().item()
        accuracy = correct / len(true_labels)
    return accuracy

def get_per_class_acc_lp(model, data, mask, og_data, num_classes, device):
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
        pred = pred_scores.argmax(dim=1)

        per_class_acc = torch.zeros(num_classes,dtype=torch.float, device=device)
        for c in range(num_classes):
            correct_in_class = ((pred == og_data.y[mask].to(pred.device)) & (og_data.y[mask].to(pred.device)==c)).sum()
            total_in_class = (og_data.y[mask] == c).sum()
            per_class_acc[c] = correct_in_class.float()/total_in_class
    return per_class_acc

    

def save_run(test_acc_NC, test_acc_LP, per_class_acc_NC, per_class_acc_LP, hidden_channels, embedding_size, num_epochs, lr_NC, lr_LP, num_seeds, two_layers, msg_split_ratio, ds, neg_pos_ratio):
    run_name = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("runs", "GAT_"+ds, run_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    np.save(os.path.join(save_path, 'test_acc_NC'), test_acc_NC)
    np.save(os.path.join(save_path, 'test_acc_LP'), test_acc_LP)
    np.save(os.path.join(save_path, 'per_class_acc_NC'), per_class_acc_NC.cpu())
    np.save(os.path.join(save_path, 'per_class_acc_LP'), per_class_acc_LP.cpu())

    config = {'hidden_channels': hidden_channels, 'embedding_size': embedding_size, 'num_epochs': num_epochs, 'lr_NC': lr_NC, 'lr_LP': lr_LP, 'num_seeds': num_seeds, 'two_layers': two_layers, 'msg_split_ratio': msg_split_ratio, 'dataset': ds, 'neg_pos_ratio': neg_pos_ratio}
    with open(os.path.join(save_path, "config.json"), "w") as outfile: 
        json.dump(config, outfile)

def main():
    args = parse_args()
    hidden_channels = args.hidden_channels
    middle_size = args.middle_size
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
    symmetric_sizes = args.symmetric_sizes


    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    seed_list = range(1, num_seeds + 1)
    dataset=None
    if ds == 'CiteSeer':
        dataset = CitationFull(root='../data/CitationFull', name='CiteSeer')
    elif ds == 'Cora_ML':
        dataset = CitationFull(root='../data/CitationFull', name='Cora_ML')
    elif ds == 'Chameleon':
        dataset = WikipediaNetwork(root='../data/chameleon', name='chameleon')
    elif ds == 'Roman_Empire':
        dataset = HeterophilousGraphDataset(root='../data/RomanEmpire', name='Roman-empire')
    elif ds == 'Squirrel':
        dataset = WikipediaNetwork(root='../data/squirrel', name='squirrel')
    else:
        print("Invalid dataset name.")
        return
    
    print("Running GAT on dataset:", ds)

    og_data = dataset[0]
    og_data = T.ToUndirected()(og_data).to(device)
    
    test_acc_NC = []
    test_acc_LP = []
    per_class_acc_NC = 0
    per_class_acc_LP = 0
    
    for seed in seed_list:
        print(f"Running for seed {seed}")
        
        setup_determinism(seed)
    
        transform = RandomNodeSplit(num_val=0.1, num_test=0.1)
        data = transform(og_data.clone()).to(device)
        num_classes = data.y.max().item() + 1
    
        model_NC = GAT_NC(hidden_channels, middle_size, embedding_size, num_classes, two_layers).to(device)
        optimizer_NC = torch.optim.Adam(model_NC.parameters(), lr=lr_NC)
        criterion_NC = torch.nn.CrossEntropyLoss()
    
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
        per_class_acc_NC += (1/num_seeds)*get_per_class_acc_nc(model_NC, data, data.test_mask, og_data, num_classes, device)

        train_data_LP, val_data_LP, test_data_LP = prepare_lp_data(data, data.train_mask, data.val_mask, data.test_mask, device, num_classes, msg_split_ratio, neg_pos_ratio)
    
        train_data_LP = train_data_LP.to(device)
        val_data_LP = val_data_LP.to(device)
        test_data_LP = test_data_LP.to(device)
    
        metadata = train_data_LP.metadata()
        model_LP = LinkPredictor(hidden_channels, middle_size, embedding_size, metadata, two_layers).to(device)
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
        per_class_acc_LP += (1/num_seeds)*get_per_class_acc_lp(model_LP, test_data_LP, data.test_mask, og_data, num_classes, device)
        
    if save_results:
        save_run(test_acc_NC, test_acc_LP, per_class_acc_NC, per_class_acc_LP, hidden_channels, embedding_size, num_epochs, lr_NC, lr_LP, num_seeds, two_layers, msg_split_ratio, ds, neg_pos_ratio)

if __name__=="__main__":
    main()