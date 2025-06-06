import os, argparse, random, json, time
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import LabelPropagation
from sklearn.metrics import average_precision_score
from preprocess_data import load_blog, load_dblp
from torch_geometric.datasets import Yelp



def parse_args():
    p = argparse.ArgumentParser("Multi-label LPA baseline")
    p.add_argument("--num_seeds", type=int, default=10)
    p.add_argument("--save_results", type=bool, default=False)
    p.add_argument("--ds", type=str, default="blog", choices=["blog", "DBLP", "Yelp"])
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.8)
    return p.parse_args()


def setup_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_dataset(name):
    if name == "blog":
        return load_blog()
    if name == "DBLP":
        return load_dblp()
    if name == "Yelp":
        return Yelp(root="data/yelp")[0]
    raise ValueError(name)


def ap_score(y_true, y_pred):
    return average_precision_score(
        y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro"
    )

def save_run(test_ap_NC, test_ap_LP, args):
    run_dir = Path("runs") / f"LPA_{args.ds}" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "test_ap_NC.npy", np.asarray(test_ap_NC))
    np.save(run_dir / "test_ap_LP.npy", np.asarray(test_ap_LP))
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.ds == 'Yelp':
        device = 'cpu'
    print("device:", device)

    og_data = load_dataset(args.ds)
    og_data = T.ToUndirected()(og_data).to(device)

    num_classes = og_data.y.max().item() + 1
    num_nodes = og_data.x.size(0)

    test_ap_NC = []
    test_ap_LP = []

    # custom split files for BLOG
    blog_splits = {
        1: "data/blog/split_0.pt",
        2: "data/blog/split_1.pt",
        3: "data/blog/split_2.pt",
    }

    num_seeds = args.num_seeds
    if args.ds == 'blog':
        num_seeds = 3

    for seed in range(1, num_seeds + 1):
        print(f"\n=== seed {seed} ===")
        setup_determinism(seed)

        data = og_data.clone()

        if args.ds == "blog":
            split_masks = torch.load(blog_splits[seed])
            data.train_mask = split_masks["train_mask"].to(device)
            data.val_mask   = split_masks["val_mask"].to(device)
            data.test_mask  = split_masks["test_mask"].to(device)
        else:
            data = RandomNodeSplit(num_val=0.1, num_test=0.1)(data).to(device)

        num_classes = data.y.size(1)

        y_input = torch.zeros_like(data.y, dtype=torch.float)
        y_input[data.train_mask] = data.y[data.train_mask]

        lpa = LabelPropagation(num_layers=args.num_layers, alpha=args.alpha).to(device)

        logits_NC = lpa(y_input, data.edge_index, mask=data.train_mask)
        
        ap_train_NC = ap_score(data.y[data.train_mask], logits_NC[data.train_mask])
        ap_val_NC   = ap_score(data.y[data.val_mask],   logits_NC[data.val_mask])
        ap_test_NC  = ap_score(data.y[data.test_mask],  logits_NC[data.test_mask])

        print(f"AP  train {ap_train_NC:.4f} | val {ap_val_NC:.4f} | test {ap_test_NC:.4f}")
        test_ap_NC.append(ap_test_NC)

        # rewire to LP

        label_node_ids = torch.arange(num_nodes, num_nodes + num_classes, dtype=torch.long).to(device)
        train_idx   = data.train_mask.nonzero(as_tuple=True)[0]
        y_train = data.y[train_idx]

        row, col = y_train.nonzero(as_tuple=True)

        src = train_idx[row]
        dst = label_node_ids[col]


        label_edges = torch.stack([src, dst], dim=0)
        label_edges_rev = label_edges.flip(0)

        data.edge_index = torch.cat([data.edge_index, label_edges, label_edges_rev], dim=1)

        pad = torch.zeros(num_classes, dtype=torch.bool, device=device)

        data.train_mask = torch.cat([data.train_mask, pad], dim=0)
        data.val_mask   = torch.cat([data.val_mask,   pad], dim=0)
        data.test_mask  = torch.cat([data.test_mask,  pad], dim=0)

        extra_y = torch.eye(num_classes, device=device, dtype=data.y.dtype)
        data.y  = torch.cat([data.y, extra_y], dim=0)

        y_input = torch.zeros_like(data.y, dtype=torch.float)
        y_input[data.train_mask] = data.y[data.train_mask]

        logits_LP = lpa(y_input, data.edge_index, mask=data.train_mask)
        
        ap_train_LP = ap_score(data.y[data.train_mask], logits_LP[data.train_mask])
        ap_val_LP   = ap_score(data.y[data.val_mask],   logits_LP[data.val_mask])
        ap_test_LP  = ap_score(data.y[data.test_mask],  logits_LP[data.test_mask])

        print(f"AP  train {ap_train_LP:.4f} | val {ap_val_LP:.4f} | test {ap_test_LP:.4f}")
        test_ap_LP.append(ap_test_LP)

    if args.save_results:
        save_run(test_ap_NC, test_ap_LP, args)
        print("saved results!")

    print("\n=== summary ===")
    print("NC mean {:.4f} ± {:.4f}".format(np.mean(test_ap_NC), np.std(test_ap_NC)))
    print("LP mean {:.4f} ± {:.4f}".format(np.mean(test_ap_LP), np.std(test_ap_LP)))


if __name__ == "__main__":
    main()