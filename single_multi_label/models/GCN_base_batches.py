import os
import argparse
import random
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import time
import json
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GraphConv, HeteroConv
from sklearn.metrics import average_precision_score
from preprocess_data import load_blog, load_dblp
from torch_geometric.datasets import Yelp
from torch_geometric.loader import NeighborLoader



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
    parser.add_argument("--batch_size", type=int, default=20, help="Amount of source nodes in each batch.")


    
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


def train_node_classification(model, data, optimizer, criterion, loader, device):
    total_loss = 0
    for batch in loader:
        batch.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        y = batch.y[:batch.batch_size]
        out = out[:batch.batch_size]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def evaluate_node_classification(model, data, mask, og_data, loader, device):
    total_precision = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            out = model(batch.x, batch.edge_index)
            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            #print(y)


            precision = ap_score(y, out)
            total_precision += precision
    return total_precision/len(loader)


def sub_data(data, mask, device):
    s_data = HeteroData()
    s_data['paper'].x = data.x
    s_data['paper'].y = data.y
    s_data['paper', 'cites', 'paper'].edge_index = data.edge_index
    num_labels = data.y.size(1)
    s_data['label'].x = torch.eye(num_labels, dtype=torch.float32, device='cpu')
    mask_indices = torch.nonzero(mask, as_tuple=False).flatten()#nodes that will be wired
    mask_y = data.y[mask_indices]
    num_edges_to_create = int(mask_y.sum().item())
    label_edges = torch.zeros((2, num_edges_to_create), dtype=torch.int64, device='cpu')
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
    neg_edge_index = torch.zeros((2, num_paper_nodes * (num_classes - 1)), device='cpu')
    neg_edge_index[0, :] = pos_edge_index[0].repeat_interleave(num_classes - 1)
    labels = torch.arange(num_classes, device='cpu')
    
    for i in range(num_paper_nodes):
        positive_label = pos_edge_index[1, i].item()
        remaining_labels = labels[labels != positive_label]
        neg_edge_index[1, i * (num_classes - 1):(i + 1) * (num_classes - 1)] = remaining_labels
    print("SHOULD NOT BE HERE")
    return neg_edge_index.to(torch.long)

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
        train_data_LP['paper', 'is', 'label'].neg_edge_index = my_negative_sampling(train_data_LP['paper', 'is', 'label'].edge_label_index, train_data_LP['paper', 'is', 'label'].edge_label_index.size(1), num_classes, 'cpu')


    train_data_LP['paper'].sup_mask = torch.zeros_like(train_mask)
    train_data_LP['paper'].sup_mask[sup_edge_index[0]] = 1

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
    
    def decode_with_embs(self, z_paper, z_label, edge_label_index):
        src, dst = edge_label_index
        return self.decoder(z_paper[src], z_label[dst])

def embed_labels(model: LinkPredictor, label_loader, num_classes, embedding_size, device):
    label_embs = torch.zeros(num_classes, embedding_size, device=device)
    model.eval()
    with torch.no_grad():
        for batch in label_loader:
            batch=batch.to(device)
            label_n_id = batch['label'].n_id
            # forward pass to get z_dict for all node types in this subgraph
            z_dict = model(batch.x_dict, batch.edge_index_dict)
            z_label_sub = z_dict['label']
            #place them into the big label_embs at the correct rows
            label_embs[label_n_id] = z_label_sub
    return label_embs

def convert_batch_edges(batch, is_training, neg_pos_ratio):
    global_to_local_paper = {gid.item(): i for i, gid in enumerate(batch['paper'].n_id)}
    global_to_local_label = {gid.item(): i for i, gid in enumerate(batch['label'].n_id)}
    if is_training and (neg_pos_ratio==-1):
        # negative
        # we only want the negative edges of the three source nodes, but we want all of them, even those that are not in the k hop subgraph!
        neg_edge_index =  batch['paper', 'is', 'label'].neg_edge_index
        paper_mask = torch.isin(neg_edge_index[0], batch['paper'].n_id[:batch['paper'].batch_size])
        #label_mask = torch.isin(neg_edge_index[1], batch['label'].n_id)           
        #final_mask = paper_mask & label_mask
        neg_edge_index = neg_edge_index[:, paper_mask]
        batch['paper', 'is', 'label'].neg_edge_index = neg_edge_index
        global_neg_edge_index_label = neg_edge_index[1].unsqueeze(0)
        local_neg_edge_index_paper = torch.tensor([[global_to_local_paper[node.item()] for node in batch['paper', 'is', 'label'].neg_edge_index[0]]], dtype=torch.long)
        #local_neg_edge_index_label = torch.tensor([[global_to_local_label[node.item()] for node in batch['paper', 'is', 'label'].neg_edge_index[1]]], dtype=torch.long)
        local_neg_edge_index = torch.cat((local_neg_edge_index_paper, global_neg_edge_index_label))
        batch['paper', 'is', 'label'].neg_edge_index = local_neg_edge_index

    # positive
    # I will keep label ids as global but translate the paper ids to local
    pos_edge_index = batch['paper', 'is', 'label'].edge_label_index
    paper_mask = torch.isin(pos_edge_index[0], batch['paper'].n_id[:batch['paper'].batch_size])
    pos_edge_index = pos_edge_index[:, paper_mask]
    
    
    local_pos_edge_index_paper = torch.tensor([[global_to_local_paper[node.item()] for node in pos_edge_index[0]]], dtype=torch.long)
    global_pos_edge_index_label = pos_edge_index[1].unsqueeze(0)

    local_pos_edge_index = torch.cat((local_pos_edge_index_paper, global_pos_edge_index_label), dim=0)
    batch['paper', 'is', 'label'].edge_label_index = local_pos_edge_index

    # uncomment to turn local label ids into global for the message passing edges as well
    #batch['paper', 'is', 'label'].edge_index[1] = batch['label'].n_id[batch['paper', 'is', 'label'].edge_index[1]]
    #batch['label', 'rev_is', 'paper'].edge_index[0] = batch['label'].n_id[batch['label', 'rev_is', 'paper'].edge_index[0]]
    

    return batch

def train_link_prediction(model, optimizer, criterion, device, paper_loader, label_loader, train_data_LP, neg_pos_ratio, num_classes, embedding_size):
    total_loss = 0
    model.train()
    
    #breakpoint()
    for batch in paper_loader:
        batch = convert_batch_edges(batch, True, neg_pos_ratio)
        batch.to(device)
        optimizer.zero_grad()
        z_dict = model(batch.x_dict, batch.edge_index_dict)
        z_paper = z_dict['paper'][:batch['paper'].batch_size]
        label_embs = embed_labels(model, label_loader, num_classes, embedding_size, device)
        pos_edge_index = batch['paper', 'is', 'label'].edge_label_index
        pos_pred = model.decode_with_embs(z_paper, label_embs, pos_edge_index)

        neg_edge_index = None
        if neg_pos_ratio > -1:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(batch['paper'].batch_size, num_classes),
                num_neg_samples=neg_pos_ratio*pos_edge_index.size(1),
                method='sparse'
            )
        else:
            neg_edge_index = batch['paper', 'is', 'label'].neg_edge_index

        neg_pred = model.decode_with_embs(z_paper, label_embs, neg_edge_index)
        pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
        neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(paper_loader)


def evaluate_link_prediction(model, device, loader, train_data_LP, neg_pos_ratio, label_loader, num_classes, embedding_size):
    model.eval()
    total_precision = 0

    label_embeddings = embed_labels(
        model, 
        label_loader, 
        num_classes, 
        embedding_size, 
        device
    )
    with torch.no_grad():
        for batch in loader:
            batch = convert_batch_edges(batch, False, neg_pos_ratio)
            batch.to(device)
            z_dict = model(batch.x_dict, batch.edge_index_dict)
            paper_embeddings = z_dict['paper'][:batch["paper"].batch_size]
            z_paper = paper_embeddings.unsqueeze(1)
            z_label = label_embeddings.unsqueeze(0)
            x = z_paper * z_label
            x = x.view(-1, x.size(-1))
            pred_scores = model.decoder.lin(x)
            pred_scores = pred_scores.view(z_paper.size(0), z_label.size(1))
            pred_probs = torch.sigmoid(pred_scores)
            pred_scores = pred_scores.view(z_paper.size(0), z_label.size(1))
            true_labels = batch['paper'].y[:batch['paper'].batch_size]
            average_precision = ap_score(true_labels, pred_probs)
            total_precision += average_precision

    return total_precision/len(loader)

    

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
    batch_size = args.batch_size


    
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
    
    test_prec_NC = []
    test_prec_LP = []
    
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

        num_neighbors_NC = [10, 5, 5]
        num_workers = 4

        loader_nc_train = NeighborLoader(
                data,
                num_neighbors=num_neighbors_NC,
                batch_size=batch_size,
                input_nodes=torch.where(data.train_mask)[0],
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True
            )
        loader_nc_val = NeighborLoader(
                data,
                num_neighbors=num_neighbors_NC,
                batch_size=batch_size,
                input_nodes=torch.where(data.val_mask)[0],
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True
            )
        loader_nc_test = NeighborLoader(
                data,
                num_neighbors=num_neighbors_NC,
                batch_size=batch_size,
                input_nodes=torch.where(data.test_mask)[0],
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True
            )
    
        model_NC = GCN_NC(hidden_channels, embedding_size, num_classes, two_layers).to(device)
        optimizer_NC = torch.optim.Adam(model_NC.parameters(), lr=lr_NC)
        criterion_NC = torch.nn.BCEWithLogitsLoss()
    
        start_patience = 100
        patience = start_patience
        best_val_nc = 0
        best_test_prec_nc = 0

        for epoch in range(1, num_epochs + 1):
            loss_nc = train_node_classification(model_NC, data, optimizer_NC, criterion_NC, loader_nc_train, device)
            train_prec = evaluate_node_classification(model_NC, data, data.train_mask, og_data, loader_nc_train, device)
            val_prec = evaluate_node_classification(model_NC, data, data.val_mask, og_data, loader_nc_val, device)
            test_prec = evaluate_node_classification(model_NC, data, data.test_mask, og_data, loader_nc_test, device)
            if print_epochs:
                print(f'Epoch: {epoch:03d}, '
                      f'Loss: {loss_nc:.4f}, '
                      f'Train AP: {train_prec:.4f}, '
                      f'Val AP: {val_prec:.4f}, '
                      f'Test AP: {test_prec:.4f}')
            if val_prec > best_val_nc:
                best_val_nc = val_prec
                best_test_prec_nc = test_prec
                patience = start_patience
            else:
                patience -= 1

            if patience <= 0:
                print('Early stopping due to no improvement in validation AP score.')
                break

        print(f"NC Test Ap score for seed {seed}: {best_test_prec_nc:.4f}")
        test_prec_NC.append(best_test_prec_nc)

        train_data_LP, val_data_LP, test_data_LP = prepare_lp_data(data, data.train_mask, data.val_mask, data.test_mask, device, num_classes, msg_split_ratio, neg_pos_ratio)

        for d in (train_data_LP, val_data_LP, test_data_LP):
            ei = d['paper', 'cites', 'paper'].edge_index
            ei, _ = add_self_loops(
                ei,
                num_nodes=d['paper'].num_nodes
            )
            d['paper', 'cites', 'paper'].edge_index = ei

        num_neighbors_LP = {key: [10, 5, 5] for key in train_data_LP.edge_types}

        label_node_ids = torch.arange(num_classes)
        
        label_loader = NeighborLoader(
            train_data_LP,
            input_nodes=('label', label_node_ids),
            num_neighbors=[10,5,5],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

        loader_lp_train = NeighborLoader(
                train_data_LP,
                num_neighbors=num_neighbors_LP,
                batch_size=batch_size,
                input_nodes=('paper', train_data_LP['paper'].sup_mask),
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True
            )
    
        loader_lp_val = NeighborLoader(
                val_data_LP,
                num_neighbors=num_neighbors_LP,
                batch_size=batch_size,
                input_nodes=('paper', data.val_mask),
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True
            )
        
        loader_lp_test = NeighborLoader(
                test_data_LP,
                num_neighbors=num_neighbors_LP,
                batch_size=batch_size,
                input_nodes=('paper', data.test_mask),
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True
            )
    
        metadata = train_data_LP.metadata()
        model_LP = LinkPredictor(hidden_channels, embedding_size, metadata, two_layers).to(device)
        optimizer_LP = torch.optim.Adam(model_LP.parameters(), lr=lr_LP)
        criterion_LP = torch.nn.BCEWithLogitsLoss()

        best_val_lp = 0
        best_test_prec_lp = 0
        patience = start_patience

        for epoch in range(1, num_epochs + 1):
            loss_lp = train_link_prediction(model_LP, optimizer_LP, criterion_LP, device, loader_lp_train, label_loader, train_data_LP, neg_pos_ratio, num_classes, embedding_size)
            train_prec = evaluate_link_prediction(model_LP, device, loader_lp_train, train_data_LP, neg_pos_ratio, label_loader, num_classes, embedding_size)
            val_prec = evaluate_link_prediction(model_LP, device, loader_lp_val, val_data_LP, neg_pos_ratio, label_loader, num_classes, embedding_size)
            test_prec = evaluate_link_prediction(model_LP, device, loader_lp_test, test_data_LP, neg_pos_ratio, label_loader, num_classes, embedding_size)
            
            if print_epochs:
                print(f'Epoch: {epoch:03d}, '
                      f'Loss: {loss_lp:.4f}, '
                      f'Train AP: {train_prec:.4f}, '
                      f'Val AP: {val_prec:.4f}, '
                      f'Test AP: {test_prec:.4f}')
            if val_prec > best_val_lp:
                best_val_lp = val_prec
                best_test_prec_lp = test_prec
                patience = start_patience
            else:
                patience -= 1

            if patience <= 0:
                print('Early stopping due to no improvement in validation AP score.')
                break

        test_prec_LP.append(best_test_prec_lp)
        print(f"LP Test AP score for seed {seed}: {best_test_prec_lp:.4f}\n")
        
    if save_results:
        save_run(test_prec_NC, test_prec_LP, hidden_channels, embedding_size, num_epochs, lr_NC, lr_LP, num_seeds, two_layers, msg_split_ratio, ds, neg_pos_ratio)

if __name__=="__main__":
    main()