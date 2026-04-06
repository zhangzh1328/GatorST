import os
import random
import torch
import numpy as np
import scanpy as sc
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


def soft_kmeans(X, K, max_iter=100, tol=1e-6):
    N, D = X.shape

    km = KMeans(n_clusters=K, random_state=0, n_init=10)
    km.fit(X)
    centroids = km.cluster_centers_.copy()

    for it in range(max_iter):
        dists_all = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2
        hard_assign = np.argmin(dists_all, axis=1)
        sigma_sq = np.zeros(K)
        for k in range(K):
            members = X[hard_assign == k]
            if len(members) > 1:
                sigma_sq[k] = np.mean(np.linalg.norm(members - centroids[k], axis=1) ** 2)
            else:
                sigma_sq[k] = 1.0
        sigma_sq = np.maximum(sigma_sq, 1e-8)

        log_numerator = -0.5 * dists_all / sigma_sq[None, :]
        log_numerator_max = log_numerator.max(axis=1, keepdims=True)
        exp_vals = np.exp(log_numerator - log_numerator_max)
        P = exp_vals / exp_vals.sum(axis=1, keepdims=True)

        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            weight_sum = P[:, k].sum()
            if weight_sum > 0:
                new_centroids[k] = (P[:, k, None] * X).sum(axis=0) / weight_sum
            else:
                new_centroids[k] = centroids[k]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    hard_labels = np.argmax(P, axis=1)

    return P, hard_labels, centroids


class CellDataset(Dataset):
    def __init__(self, X, y_true, y_pseudo, soft_labels, loc, subgraphs):
        self.X = X
        self.y_true = y_true
        self.y_pseudo = y_pseudo
        self.soft_labels = soft_labels
        self.loc = loc
        self.subgraphs = subgraphs

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, index):
        return (self.X[index], self.y_true[index], self.y_pseudo[index],
                self.soft_labels[index], self.loc[index], self.subgraphs[index])


def collate_fn(batch):
    batch_x = torch.stack([item[0] for item in batch])
    batch_y_true = torch.stack([item[1] for item in batch])
    batch_y_pseudo = torch.stack([item[2] for item in batch])
    batch_soft = torch.stack([item[3] for item in batch])
    batch_loc = torch.stack([item[4] for item in batch])
    batch_subgraph = [item[5] for item in batch]
    return batch_x, batch_y_true, batch_y_pseudo, batch_soft, batch_loc, batch_subgraph


def loader_construction(data_name, data_path, batch_size, k_neighbors=3,
                        n_pseudo_clusters=20, max_neighbors=20,
                        device='cuda', num_workers=4):
    data = sc.read_h5ad(data_path)
    X_all = data.X.toarray()
    y_all = np.array(data.obs.loc[:, 'ground_truth'])
    loc = data.obsm['spatial']
    
    label_encoder = LabelEncoder()
    y_all_true = label_encoder.fit_transform(y_all)
    n_clusters = len(np.unique(y_all_true))
    # X_all = PCA(n_components=200, random_state=42).fit_transform(X_all)
    
    K = n_pseudo_clusters
    soft_probs, y_pseudo, centroids = soft_kmeans(X_all, K)
    
    print(f"Building KNN graph on spatial locations")
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
    knn.fit(loc)
    knn_graph_matrix = knn.kneighbors_graph(loc, mode='connectivity')

    G = nx.Graph()
    G.add_nodes_from(range(X_all.shape[0]))
    rows, cols = knn_graph_matrix.nonzero()
    for i, j in zip(rows, cols):
        if i != j:
            G.add_edge(i, j)
    print(f"Spatial KNN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    subgraph_data_list = []
    
    for node in tqdm(G.nodes, desc="Extracting 2-hop subgraphs"):
        one_hop_neighbors = set(nx.neighbors(G, node))
        two_hop_neighbors = set()
        for neighbor in one_hop_neighbors:
            two_hop_neighbors.update(nx.neighbors(G, neighbor))

        all_neighbors = one_hop_neighbors | two_hop_neighbors
        all_neighbors.discard(node)

        if len(all_neighbors) > max_neighbors:
            one_hop_list = list(one_hop_neighbors)
            pure_two_hop = list(all_neighbors - one_hop_neighbors)
            remaining = max_neighbors - len(one_hop_list)
            if remaining > 0:
                sampled_two_hop = random.sample(pure_two_hop, min(len(pure_two_hop), remaining))
                selected = one_hop_list + sampled_two_hop
            else:
                selected = random.sample(one_hop_list, max_neighbors)
            subgraph_nodes = [node] + selected
        else:
            subgraph_nodes = [node] + list(all_neighbors)

        subgraph_nodes = list(set(subgraph_nodes))
        subgraph = G.subgraph(subgraph_nodes)

        node_mapping = {n: i for i, n in enumerate(subgraph_nodes)}
        edge_list = [[node_mapping[u], node_mapping[v]] for u, v in subgraph.edges]
        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        x = X_all[subgraph_nodes]
        data_dict = {'x': torch.tensor(x).float().cuda(), 'edge_index': edge_index.cuda()}
        subgraph_data_list.append(data_dict)
    
    X_all_t = torch.tensor(X_all).float()
    y_true_t = torch.tensor(y_all_true).long()
    y_pseudo_t = torch.tensor(y_pseudo).long()
    soft_labels_t = torch.tensor(soft_probs).float()
    loc_t = torch.tensor(loc).float()

    input_dim = X_all_t.shape[1]

    indices = np.arange(len(y_true_t))
    idx_train, idx_val = train_test_split(indices, test_size=0.2, random_state=1)
    idx_val, idx_test = train_test_split(idx_val, test_size=0.5, random_state=1)

    def make_dataset(idx):
        return CellDataset(
            X_all_t[idx].cuda(), y_true_t[idx].cuda(), y_pseudo_t[idx].cuda(),
            soft_labels_t[idx].cuda(), loc_t[idx].cuda(),
            [subgraph_data_list[i] for i in idx]
        )

    train_set = make_dataset(idx_train)
    val_set = make_dataset(idx_val)
    test_set = make_dataset(idx_test)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, input_dim, n_clusters
