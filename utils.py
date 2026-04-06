import os
import time
import random
import torch
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score


def format_time(seconds):
    if seconds <= 60:
        return '%.1fs' % seconds
    elif seconds <= 3600:
        return '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        return '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(y_true, y_pred):
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    return acc, nmi, ari, homo, comp, purity


def l1_distance(imputed_data, original_data):

    return np.mean(np.abs(original_data-imputed_data))


def rmse_f(imputed_data, original_data):
    
    return np.sqrt(np.mean((original_data - imputed_data) ** 2))


def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr

def evaluate_imp(imputed_data, original_data):
    l1 = l1_distance(imputed_data, original_data)
    rmse = rmse_f(imputed_data, original_data)
    pcc = pearson_corr(imputed_data, original_data)
    return pcc, l1, rmse
