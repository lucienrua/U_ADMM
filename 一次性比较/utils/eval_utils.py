import numpy as np
from scipy.stats import pearsonr, kendalltau

def evaluate_ranking_accuracy(X_list, Y_list, theta_hat, quantiles):
    X_all = np.vstack(X_list)
    Y_all = np.concatenate(Y_list)
    
    n = X_all.shape[0]
    from itertools import combinations
    ii, jj = map(np.array, zip(*combinations(range(n), 2)))
    
    mask = Y_all[ii] != Y_all[jj]
    ii, jj = ii[mask], jj[mask]
    
    true_sign = np.sign(Y_all[ii] - Y_all[jj])
    scores = X_all @ theta_hat.flatten()
    pred_sign = np.sign(scores[ii] - scores[jj])
    
    correct = np.sum(true_sign == pred_sign)
    total = len(true_sign)
    
    return {
        'Total_Pairs': int(total),
        'Correct_Pairs': int(correct),
        'Pairwise_Correlation': float(correct / total) if total > 0 else 0.0
    }

def evaluate_correlation(X_list, theta_true, theta_hat):
    X_all = np.vstack(X_list)
    true_scores = X_all @ theta_true.flatten()
    pred_scores = X_all @ theta_hat.flatten()
    
    pearson_corr, _ = pearsonr(true_scores, pred_scores)
    kendall_corr, _ = kendalltau(true_scores, pred_scores)
    
    return {
        'Pearson_Corr': float(pearson_corr),
        'Kendall_Corr': float(kendall_corr)
    }

def calculate_metrics(theta_true, theta_hat, threshold=1e-4):
    rmse = np.linalg.norm(theta_true - theta_hat)
    mae = np.mean(np.abs(theta_true - theta_hat))
    
    true_nonzero = np.abs(theta_true) > 0
    pred_nonzero = np.abs(theta_hat) > threshold
    
    TP = np.sum(true_nonzero & pred_nonzero)
    FP = np.sum(~true_nonzero & pred_nonzero)
    TN = np.sum(~true_nonzero & ~pred_nonzero)
    FN = np.sum(true_nonzero & ~pred_nonzero)
    
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'Selection_Acc': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1_Score': float(f1_score)
    }
