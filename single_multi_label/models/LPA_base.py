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
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--save_results", type=bool, default=False)
    p.add_argument("--ds", type=str, default="blog", choices=["blog", "DBLP", "Yelp"])
    p.add_argument("--num_layers", type=int, default=15)
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

def save_run(test_ap_nc, args):
    run_dir = Path("runs") / f"LPA_{args.ds}" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "test_ap_NC.npy", np.asarray(test_ap_nc))
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_full = load_dataset(args.ds)
    data_full = T.ToUndirected()(data_full).to(device)

    test_ap_all = []

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

        data = data_full.clone()

        if args.ds == "blog":
            split_masks = torch.load(blog_splits[seed])
            data.train_mask = split_masks["train_mask"].to(device)
            data.val_mask   = split_masks["val_mask"].to(device)
            data.test_mask  = split_masks["test_mask"].to(device)
        else:
            data = RandomNodeSplit(num_val=0.1, num_test=0.1)(data)

        num_classes = data.y.size(1)

        y_input = torch.zeros_like(data.y, dtype=torch.float)
        y_input[data.train_mask] = data.y[data.train_mask]

        lpa = LabelPropagation(num_layers=args.num_layers,
                               alpha=args.alpha).to(device)

        logits = lpa(y_input, data.edge_index, mask=data.train_mask)
        
        ap_train = ap_score(data.y[data.train_mask], logits[data.train_mask])
        ap_val   = ap_score(data.y[data.val_mask],   logits[data.val_mask])
        ap_test  = ap_score(data.y[data.test_mask],  logits[data.test_mask])

        print(f"AP  train {ap_train:.4f} | val {ap_val:.4f} | test {ap_test:.4f}")
        test_ap_all.append(ap_test)

    if args.save_results:
        save_run(test_ap_all, args)
        print("saved results to runs/LPA")

    print("\n=== summary ===")
    print("AP mean {:.4f} ± {:.4f}".format(np.mean(test_ap_all), np.std(test_ap_all)))


if __name__ == "__main__":
    main()