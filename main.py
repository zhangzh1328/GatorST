import os
import json
import time
import argparse
from data_loader import loader_construction
from model import train, test
from utils import setup_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--N_way', type=int, default=5)
    parser.add_argument('--M_shot', type=int, default=3)
    parser.add_argument('--Q_query', type=int, default=5)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--k_neighbors', type=int, default=3)
    parser.add_argument('--n_pseudo_clusters', type=int, default=4)
    args = parser.parse_args()
    
    data_folder = "./data"
    save_results_path = f"./result.json"
    save_model_folder = f"./saved_models"

    if not os.path.exists(save_model_folder):
        os.makedirs(save_model_folder)

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    alpha = args.alpha
    N_way = args.N_way
    M_shot = args.M_shot
    Q_query = args.Q_query
    tau = args.tau
    k_neighbors = args.k_neighbors
    n_pseudo_clusters = args.n_pseudo_clusters
    
    all_results = {}

    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".h5ad"):
            continue

        data_name = os.path.splitext(file_name)[0]
        
        if data_name in all_results:
            continue
        
        print(f'Start Running {data_name}')
        
        data_path = os.path.join(data_folder, file_name)

        train_loader, val_loader, test_loader, input_dim, n_clusters = \
            loader_construction(data_name, data_path, batch_size,
                                k_neighbors=k_neighbors,
                                n_pseudo_clusters=n_pseudo_clusters)
        
        for run in range(10):
            seed = run
            start_time = time.time()
            save_model_path = os.path.join(save_model_folder,
                                           f"{data_name}_model_run_{run}")
            
            best_epoch, min_loss = train(
                train_loader, val_loader, test_loader, lr=lr, seed=seed,
                epochs=epochs, n_clusters=n_clusters, input_dim=input_dim,
                save_model_path=save_model_path, alpha=alpha, N_way=N_way, 
                M_shot=M_shot, Q_query=Q_query, tau=tau)
            
            elapsed_time = time.time() - start_time
            
            results = test(test_loader, n_clusters=n_clusters, input_dim=input_dim,
                           save_model_path=save_model_path, seed=seed)

            if data_name not in all_results:
                all_results[data_name] = []
            all_results[data_name].append(results)

        with open(save_results_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)

        print(f"Results saved to {save_results_path}")
