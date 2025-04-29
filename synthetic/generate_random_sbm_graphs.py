import argparse
import random

import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A simple script with arguments")
    parser.add_argument('--seed', default=42, type=int, help='Seed for SBM graph')
    parser.add_argument('--p_same_label', default=0.05, type=float, help='Probability of edge within same label blocks')
    parser.add_argument('--p_diff_label', default=0.2, type=float, help='Probability of edge between different label blocks')
    parser.add_argument('--block_size', default=50, type=int, help='Size of blocks in the SBM')

    args = parser.parse_args()
    s = args.seed
    p_same_label = args.p_same_label
    p_diff_label = args.p_diff_label
    block_size = args.block_size

    torch.manual_seed(s)
    random.seed(s)

    # Settings
    n_blocks = 4
    block_sizes = [block_size, block_size, block_size, block_size]  # total 200 nodes
    label_mapping = {0: 0, 1: 0, 2: 1, 3: 1}  # blocks 0/1 -> label 0, blocks 2/3 -> label 1

    probs = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i == j:
                probs[i, j] = p_same_label
            else:
                if label_mapping[i] == label_mapping[j]:
                    probs[i, j] = p_same_label
                else:
                    probs[i, j] = p_diff_label


    # Build the stochastic block model graph
    G = nx.stochastic_block_model(block_sizes, probs)

    # Noise level: fraction of nodes whose labels will be flipped
    noise_level = 0.1  # 10% noisy labels

    # Assign labels with noise
    labels = {}
    for node, data in G.nodes(data=True):
        block = data['block']
        true_label = label_mapping[block]
        if random.random() < noise_level:
            # Flip label with probability = noise_level
            noisy_label = 1 - true_label
            labels[node] = noisy_label
        else:
            labels[node] = true_label

    # Add labels to graph
    nx.set_node_attributes(G, labels, 'label')

    # --- Optional: Evaluate heterophily level ---
    same_label_edges = 0
    diff_label_edges = 0
    for u, v in G.edges():
        if labels[u] == labels[v]:
            same_label_edges += 1
        else:
            diff_label_edges += 1

    total_edges = same_label_edges + diff_label_edges
    print(f"Same label edges: {same_label_edges} ({100 * same_label_edges / total_edges:.2f}%)")
    print(f"Different label edges: {diff_label_edges} ({100 * diff_label_edges / total_edges:.2f}%)")

    g_tensor = from_networkx(G)
    g_tensor.y = g_tensor.label
    g_tensor.x = torch.rand(g_tensor.num_nodes, 128)

    breakpoint()

    # save the tensor
    torch.save(g_tensor, f"sbm_{str(p_same_label)}_{str(p_diff_label)}_{s}.pt")




