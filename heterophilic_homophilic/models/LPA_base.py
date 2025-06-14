import os
import argparse
import random
import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import HeterophilousGraphDataset

from torch_geometric.nn import LabelPropagation
import time
import json


def parse_args():
    parser = argparse.ArgumentParser(description="NC/LP Model Hyperparameters")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds to run (starting from 1).")
    parser.add_argument("--save_results", type=bool, default=False, help="Set to True if results and config should be saved in runs dir)")
    parser.add_argument("--ds", type=str, default='Cora_ML', help="Which dataset to run on.")
    parser.add_argument("--num_layers", type=int, default=3,help="# propagation steps (depth)")
    parser.add_argument("--alpha", type=float, default=0.8,help="restart probability (1-alpha)")

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

def accuracy(pred, true, mask):
    return float((pred[mask] == true[mask]).sum()) / int(mask.sum())

def save_run(test_acc_NC, test_acc_LP, args):
    run_name  = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("runs", f"LPA_{args.ds}", run_name)
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "test_acc_NC.npy"), np.asarray(test_acc_NC))
    np.save(os.path.join(save_path, "test_acc_LP.npy"), np.asarray(test_acc_LP))

    cfg_path = os.path.join(save_path, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

def main():
    args = parse_args()
    num_seeds = args.num_seeds
    save_results = args.save_results
    ds = args.ds
    num_layers = args.num_layers
    alpha = args.alpha

    
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
    elif ds == 'OGBN':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../data/ogbn_arxiv')
    else:
        print("Invalid dataset name.")
        return
    
    print("Running LPA on dataset:", ds)

    og_data = dataset[0]
    og_data = T.ToUndirected()(og_data).to(device)

    num_classes = og_data.y.max().item() + 1
    num_nodes = og_data.x.size(0)
    
    test_acc_NC = []
    test_acc_LP = []
    
    for seed in seed_list:
        print(f"Running for seed {seed}")
        
        setup_determinism(seed)
    
        transform = RandomNodeSplit(num_val=0.1, num_test=0.1)
        data = transform(og_data.clone()).to(device)

        if data.y.dim() == 2 and data.y.size(1) == 1:
            data.y = data.y.squeeze(1)

        print(data)
        print(data.train_mask)
    
        lpa = LabelPropagation(num_layers=num_layers, alpha=alpha).to(device)
        
        logits_NC = lpa(data.y, data.edge_index, mask=data.train_mask)
        preds_NC = logits_NC.argmax(dim=-1)

        acc_train_NC = accuracy(preds_NC, data.y, data.train_mask)
        acc_val_NC   = accuracy(preds_NC, data.y, data.val_mask)
        acc_test_NC  = accuracy(preds_NC, data.y, data.test_mask)
        print(f"NC  train {acc_train_NC:.4f} | val {acc_val_NC:.4f} | "
              f"test {acc_test_NC:.4f}")
        test_acc_NC.append(acc_test_NC)
        

        # rewire to LP
        
        label_node_ids = torch.arange(num_nodes, num_nodes + num_classes, dtype=torch.long)
        num_train = data.train_mask.sum().item()
        
        train_label_edges = torch.zeros((2, num_train), dtype=torch.int64)
        train_label_edges[0] = data.train_mask.nonzero(as_tuple=True)[0]
        if ds == 'OGBN':
            train_label_edges[1] = data.y[data.train_mask].squeeze() + num_nodes
        else:
            train_label_edges[1] = data.y[data.train_mask] + num_nodes

        data.edge_index = torch.cat([data.edge_index, train_label_edges.to(device)], dim=1)
        data.edge_index = torch.cat([data.edge_index, train_label_edges.flip(0).to(device)], dim=1) #undirected

        label_nodes_mask = torch.zeros(num_classes, dtype=torch.bool).to(device)
        data.train_mask = torch.cat((data.train_mask, label_nodes_mask)) # change to ones here if label nodes should be included
        data.val_mask = torch.cat((data.val_mask, label_nodes_mask))
        data.test_mask = torch.cat((data.test_mask, label_nodes_mask))

        data.y = torch.cat((data.y, torch.arange(0, num_classes).to(device)))

        logits_LP = lpa(data.y, data.edge_index, mask=data.train_mask)
        preds_LP = logits_LP.argmax(dim=-1)

        acc_train_LP = accuracy(preds_LP, data.y, data.train_mask)
        acc_val_LP   = accuracy(preds_LP, data.y, data.val_mask)
        acc_test_LP  = accuracy(preds_LP, data.y, data.test_mask)
        print(f"LP  train {acc_train_LP:.4f} | val {acc_val_LP:.4f} | "
              f"test {acc_test_LP:.4f}")
        test_acc_LP.append(acc_test_LP)
    
    if save_results:
        save_run(test_acc_NC, test_acc_LP, args)
        print("saved results!")
    
    print("NC  mean {:.4f} +- {:.4f}".format(np.mean(test_acc_NC),  np.std(test_acc_NC)))
    print("LP  mean {:.4f} +- {:.4f}".format(np.mean(test_acc_LP),  np.std(test_acc_LP)))

if __name__=="__main__":
    main()
