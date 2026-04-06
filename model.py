import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import * 
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm


class MetaContrastiveLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(MetaContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings, labels):
        unique_classes = torch.unique(labels)
        N = len(unique_classes)
        if N < 2:
            return torch.tensor(0.0, device=embeddings.device)

        prototypes = []
        class_masks = []
        for cls in unique_classes:
            mask = (labels == cls)
            class_masks.append(mask)
            proto = embeddings[mask].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)

        total_loss = 0.0
        total_count = 0

        for cls_idx, cls in enumerate(unique_classes):
            mask = class_masks[cls_idx]
            cls_embeddings = embeddings[mask]
            M_i = cls_embeddings.shape[0]

            sims = torch.matmul(cls_embeddings, prototypes.T) / self.tau
            pos_sims = sims[:, cls_idx]
            log_denom = torch.logsumexp(sims, dim=1)
            sample_loss = -(pos_sims - log_denom)

            total_loss += sample_loss.sum()
            total_count += M_i

        if total_count == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return total_loss / total_count


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[128, 256]):
        super(Decoder, self).__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class Model(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=128, output_dim=10):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        self.w_imp = Decoder(hidden_dim, input_dim, hidden_dims=[128, 256])

        self.mae_loss = torch.nn.L1Loss(reduction='mean')
        self.l1loss = torch.nn.L1Loss(reduction='none')

    def GCN(self, batch):
        embs = []
        for data in batch:
            x = self.conv1(data['x'], data['edge_index'])
            x = F.relu(x)
            x = self.conv2(x, data['edge_index'])
            embs.append(x.mean(0))
        return torch.stack(embs, 0)

    def forward(self, x, batch_subgraph):
        z = self.encoder(x)
        node_emb = self.GCN(batch_subgraph)
        x_imp = self.w_imp(node_emb)
        combined = torch.cat((z, node_emb), dim=1)

        return combined, x_imp


def sample_episode(dataset, N_way, M_shot, Q_query):
    labels_np = dataset.y_pseudo.cpu().numpy()
    unique_classes, counts = np.unique(labels_np, return_counts=True)

    min_samples = 2
    valid_mask = counts >= min_samples
    valid_classes = unique_classes[valid_mask]

    N = min(N_way, len(valid_classes))
    if N < 2:
        return None

    selected_classes = np.random.choice(valid_classes, size=N, replace=False)

    support_indices = []
    query_indices = []

    for cls in selected_classes:
        cls_indices = np.where(labels_np == cls)[0]
        np.random.shuffle(cls_indices)

        n_available = len(cls_indices)
        m = min(M_shot, n_available - 1)
        if m < 1:
            m = 1
        q = min(Q_query, n_available - m)
        if q < 1:
            continue

        support_indices.extend(cls_indices[:m].tolist())
        query_indices.extend(cls_indices[m:m + q].tolist())

    if len(support_indices) < 2 or len(query_indices) < 1:
        return None

    support_X = dataset.X[support_indices]
    support_y = dataset.y_pseudo[support_indices]
    support_soft = dataset.soft_labels[support_indices]
    support_subgraphs = [dataset.subgraphs[i] for i in support_indices]

    query_X = dataset.X[query_indices]
    query_y = dataset.y_pseudo[query_indices]
    query_soft = dataset.soft_labels[query_indices]
    query_subgraphs = [dataset.subgraphs[i] for i in query_indices]

    return (support_X, support_y, support_soft, support_subgraphs,
            query_X, query_y, query_soft, query_subgraphs)


def train(train_loader, valid_loader, test_loader, lr, seed, epochs,
          n_clusters, input_dim, save_model_path, alpha=0.5, 
          N_way=5, M_shot=5, Q_query=5, tau=1.0, device='cuda'):

    model = Model(input_dim, output_dim=n_clusters).cuda()
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    meta_contrastive_loss_fn = MetaContrastiveLoss(tau=tau)
    ce_loss_fn = nn.CrossEntropyLoss()

    setup_seed(seed)
    train_loss_history = []
    valid_loss_history = []
    best_epoch = 0

    train_dataset = train_loader.dataset
    val_dataset = valid_loader.dataset

    steps = len(train_loader)
    min_loss = 1e6
    patience = 20
    patience_counter = 0
    
    for each_epoch in range(epochs):
        if patience_counter >= patience:
            print(f"Early stopping at epoch {each_epoch + 1}")
            break

        batch_losses = []
        model.train()

        with tqdm(total=steps, desc=f'Epoch {each_epoch + 1}/{epochs}', unit='batch') as pbar:
            for step, (batch_x, batch_y_true, batch_y_pseudo, batch_soft,
                        batch_loc, batch_subgraph) in enumerate(train_loader):
                
                episode = sample_episode(
                    train_dataset, N_way=N_way, M_shot=M_shot, Q_query=Q_query
                )

                if episode is not None:
                    (s_X, s_y, s_soft, s_subgraphs,
                     q_X, q_y, q_soft, q_subgraphs) = episode

                    s_combined, _ = model(s_X, s_subgraphs)
                    loss_ct = meta_contrastive_loss_fn(s_combined, s_y)

                    q_combined, _ = model(q_X, q_subgraphs)
                    q_logits = model.classifier(q_combined)
                    loss_ce = ce_loss_fn(q_logits, q_y)

                    loss_meta = alpha * loss_ct + (1 - alpha) * loss_ce

                else:
                    loss_meta = torch.tensor(0.0, device=device)

                loss = loss_meta 

                opt_model.zero_grad()
                loss.backward()
                opt_model.step()

                batch_losses.append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        train_loss_history.append(np.mean(batch_losses))
        
        with torch.no_grad():
            val_losses = []
            model.eval()
            for step, (batch_x, batch_y_true, batch_y_pseudo, batch_soft,
                        batch_loc, batch_subgraph) in enumerate(valid_loader):
                
                episode = sample_episode(
                    val_dataset, N_way=N_way, M_shot=M_shot, Q_query=Q_query
                )
                if episode is not None:
                    (s_X, s_y, s_soft, s_subgraphs,
                     q_X, q_y, q_soft, q_subgraphs) = episode
                    s_combined, _ = model(s_X, s_subgraphs)
                    loss_ct = meta_contrastive_loss_fn(s_combined, s_y)
                    q_combined, _ = model(q_X, q_subgraphs)
                    q_logits = model.classifier(q_combined)
                    loss_ce = ce_loss_fn(q_logits, q_y)
                    loss_meta = alpha * loss_ct + (1 - alpha) * loss_ce
                else:
                    loss_meta = torch.tensor(0.0, device=device)

                val_loss = loss_meta 
                val_losses.append(val_loss.item())
        
        valid_loss_history.append(np.mean(val_losses))
        cur_loss = valid_loss_history[-1]

        print(f"Epoch {each_epoch+1}: train_loss={train_loss_history[-1]:.4f}, "
              f"val_loss={cur_loss:.4f}")

        if cur_loss < min_loss or each_epoch == 0:
            print(f'Saving model at Epoch {each_epoch} with loss {cur_loss:.4f}')
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict()
            }
            torch.save(state, save_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

    return best_epoch, min_loss


def test(test_loader,
         n_clusters,
         input_dim,
         save_model_path,
         seed,
         task=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(input_dim, output_dim=n_clusters).to(device)
    ckpt = torch.load(save_model_path, map_location=device, weights_only=True)
    weights = ckpt['net']
    model.load_state_dict(weights)
    model.eval()
    
    z_test = []
    y_test = []
    x_test = []
    x_imp_test = []
    for step, (batch_x, batch_y_true, batch_y_pseudo, batch_soft,
                batch_loc, batch_subgraph) in enumerate(test_loader):
        combined, x_imp = model(batch_x, batch_subgraph)
        z_test.append(combined.cpu().detach().numpy())
        y_test.append(batch_y_true.cpu().detach().numpy())
        x_test.append(batch_x.cpu().detach().numpy())
        x_imp_test.append(x_imp.cpu().detach().numpy())
        
    z_test = np.vstack(z_test)
    y_test = np.hstack(y_test)
    x_test = np.vstack(x_test)
    x_imp_test = np.vstack(x_imp_test)

    if task == 'imputation':
        pcc, l1, rmse = evaluate_imp(x_imp_test, x_test)
        results_imp = {
            'PCC': float(pcc),
            'L1': float(l1),
            'RMSE': float(rmse)
        }
        return results_imp
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(z_test)  
    y_kmeans_test = kmeans.labels_

    acc, nmi, ari, homo, comp, purity = evaluate(y_test, y_kmeans_test)
    
    results = {
        'ACC': float(acc),
        'ARI': float(ari),
        'NMI': float(nmi),
        'Purity': float(purity),
        'Homogeneity': float(homo)
    }
    
    return results
