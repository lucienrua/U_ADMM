import numpy as np

def evaluate_ranking_accuracy(X_list, Y_list, theta_hat, quantiles):
    X_all = np.vstack(X_list)
    Y_all = np.concatenate(Y_list)
    
    # Calculate pairwise accuracy
    n = X_all.shape[0]
    # To avoid memory issues with large n, we can sample pairs or calculate efficiently
    # For n=1000 (10 nodes * 100 samples), C(1000, 2) is ~500,000, which is manageable
    
    from itertools import combinations
    ii, jj = map(np.array, zip(*combinations(range(n), 2)))
    
    # Only consider pairs with different true labels
    mask = Y_all[ii] != Y_all[jj]
    ii, jj = ii[mask], jj[mask]
    
    # True pairwise relationship
    true_sign = np.sign(Y_all[ii] - Y_all[jj])
    
    # Predicted pairwise relationship
    scores = X_all @ theta_hat.flatten()
    pred_sign = np.sign(scores[ii] - scores[jj])
    
    # Calculate accuracy
    correct = np.sum(true_sign == pred_sign)
    total = len(true_sign)
    
    results = {
        'Total_Pairs': int(total),
        'Correct_Pairs': int(correct),
        'Pairwise_Accuracy': float(correct / total) if total > 0 else 0.0
    }
    
    return results

def calculate_metrics(theta_true, theta_hat):
    # 论文公式 (31): sqrt(sum((theta_hat - theta_true)^2))，即 L2 范数
    rmse = np.linalg.norm(theta_true - theta_hat)
    mae = np.mean(np.abs(theta_true - theta_hat))
    return float(rmse), float(mae)
