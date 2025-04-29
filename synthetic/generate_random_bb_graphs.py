import argparse
import random

import networkx as nx
import torch
from torch_geometric.utils import from_networkx

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A simple script with arguments")
    parser.add_argument('--m1', default=100, type=int, help='Size of the cliques')
    parser.add_argument('--m2', default=10, type=int, help='Length of the path')
    parser.add_argument('--labelling', default="homo", type=str, help='Type of labelling: homo or homosplit')
    parser.add_argument('--seed', default=42, type=int, help='Seed for noise labels')

    args = parser.parse_args()
    m1 = args.m1
    m2 = args.m2
    labelling = args.labelling

    G = nx.barbell_graph(m1, m2)
    noise_level = 0.1
    labels = {}

    # Assign labels to nodes
    if labelling == "homo":
        for node in G.nodes():
            if node <= m1-1:
                # Left clique
                true_label = 0
            elif m1 <= node <= m1+m2-1:
                # Path nodes
                if node <= m1+(m2/2)-1:
                    true_label = 0
                else:
                    true_label = 1
            else:
                # Right clique
                true_label = 1

            # Introduce noise
            if random.random() < noise_level:
                noisy_label = 1 - true_label
                labels[node] = noisy_label
            else:
                labels[node] = true_label
    else: # labelling == "homosplit"
        for node in G.nodes():
            if node <= m1-1:
                # Left clique
                true_label = 0
            elif m1 <= node <= m1+m2-1:
                # Path nodes
                    true_label = 1
            else:
                # Right clique
                true_label = 0
            # Introduce noise
            if random.random() < noise_level:
                noisy_label = 1 - true_label
                labels[node] = noisy_label
            else:
                labels[node] = true_label


    # Assign labels to graph
    nx.set_node_attributes(G, labels, 'label')

    print(f"Total nodes: {G.number_of_nodes()}")
    g_tensor = from_networkx(G)
    g_tensor.y = g_tensor.label
    g_tensor.x = torch.eye(g_tensor.num_nodes)

    # save the tensor
    torch.save(g_tensor, f"barbell_{m1}_{m2}_{labelling}.pt")

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

