import time
import numpy as np
from models.ranking import generate_ranking_data
from models.aft import generate_aft_data
from algorithms.admm import run_u_admm, init_all_nodes
from algorithms.baselines import run_global_u_erm, run_dgd
from utils.eval_utils import evaluate_ranking_accuracy, calculate_metrics

def get_metrics_ranking(theta, theta_true, X, Y, quantiles, t_cost):
    rmse, mae = calculate_metrics(theta_true, theta)
    acc_dict = evaluate_ranking_accuracy(X, Y, theta, quantiles)
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'Accuracy': float(acc_dict['Pairwise_Accuracy']),
        'Total_Pairs': acc_dict['Total_Pairs'],
        'Correct_Pairs': acc_dict['Correct_Pairs'],
        'Time': float(t_cost)
    }

def get_metrics_aft(theta, theta_true, t_cost):
    rmse, mae = calculate_metrics(theta_true, theta)
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'Time': float(t_cost)
    }

def run_single_ranking(seed, params):
    np.random.seed(seed)
    d_rank = generate_ranking_data(
        m=params['m'], n=params['n'], p_prime=params['p_prime'], 
        p=params['p'], pc=params['pc'], noise_type=params['noise_type'], rng_seed=seed
    )
    
    theta_true = d_rank['theta_true']
    X, Y, quantiles = d_rank['X'], d_rank['Y'], d_rank['quantiles']
    
    result = {'seed': seed, 'noise_type': params['noise_type']}
    
    if params.get('run_proposed', True):
        t0 = time.time()
        theta_u_r, theta_n_r, hist_r = run_u_admm(
            d_rank, T=params['T'], W_inner=params['W_inner'], 
            rho=params['rho'], lam_t=params['lam_t'], verbose=False
        )
        t_uadmm = time.time() - t0
        result['Proposed'] = get_metrics_ranking(theta_u_r[0], theta_true, X, Y, quantiles, t_uadmm)
        result['Proposed']['hist_rmse'] = hist_r['rmse']
        result['Proposed']['theta_hat'] = theta_u_r[0].flatten().tolist()
        
        # Avg MR is a byproduct of Proposed initialization
        result['Avg'] = get_metrics_ranking(theta_n_r, theta_true, X, Y, quantiles, 0.0)
        
    if params.get('run_local', True):
        t0 = time.time()
        theta0_list, _ = init_all_nodes(d_rank)
        t_local = time.time() - t0
        local_rmses, local_maes, local_accs = [], [], []
        for th in theta0_list:
            m_dict = get_metrics_ranking(th, theta_true, X, Y, quantiles, 0)
            local_rmses.append(m_dict['RMSE'])
            local_maes.append(m_dict['MAE'])
            local_accs.append(m_dict['Accuracy'])
        result['Local'] = {
            'RMSE': float(np.mean(local_rmses)),
            'MAE': float(np.mean(local_maes)),
            'Accuracy': float(np.mean(local_accs)),
            'Time': float(t_local)
        }
        
    if params.get('run_pooled', True):
        t0 = time.time()
        theta_global = run_global_u_erm(d_rank)
        t_global = time.time() - t0
        result['Pooled'] = get_metrics_ranking(theta_global, theta_true, X, Y, quantiles, t_global)
        
    if params.get('run_dgd', True):
        t0 = time.time()
        # D-subGD iterations = T * W_inner to ensure fair computation comparison
        theta_dgd = run_dgd(d_rank, T=params['T'] * params['W_inner'], lr=0.1)
        t_dgd = time.time() - t0
        result['D-subGD'] = get_metrics_ranking(theta_dgd, theta_true, X, Y, quantiles, t_dgd)
        
    return result

def run_single_aft(seed, params):
    np.random.seed(seed)
    d_aft = generate_aft_data(
        m=params['m'], n=params['n'], p=params['p'], 
        pc=params['pc'], noise_type=params['noise_type'], rng_seed=seed
    )
    
    theta_true = d_aft['theta_true']
    
    result = {'seed': seed, 'noise_type': params['noise_type']}
    
    if params.get('run_proposed', True):
        t0 = time.time()
        theta_u_a, theta_n_a, hist_a = run_u_admm(
            d_aft, T=params['T'], W_inner=params['W_inner'], 
            rho=params['rho'], lam_t=params['lam_t'], verbose=False
        )
        t_uadmm = time.time() - t0
        result['Proposed'] = get_metrics_aft(theta_u_a[0], theta_true, t_uadmm)
        result['Proposed']['hist_rmse'] = hist_a['rmse']
        result['Proposed']['theta_hat'] = theta_u_a[0].flatten().tolist()
        
        result['Avg'] = get_metrics_aft(theta_n_a, theta_true, 0.0)
        
    if params.get('run_local', True):
        t0 = time.time()
        theta0_list, _ = init_all_nodes(d_aft)
        t_local = time.time() - t0
        local_rmses, local_maes = [], []
        for th in theta0_list:
            m_dict = get_metrics_aft(th, theta_true, 0)
            local_rmses.append(m_dict['RMSE'])
            local_maes.append(m_dict['MAE'])
        result['Local'] = {
            'RMSE': float(np.mean(local_rmses)),
            'MAE': float(np.mean(local_maes)),
            'Time': float(t_local)
        }
        
    if params.get('run_pooled', True):
        t0 = time.time()
        theta_global = run_global_u_erm(d_aft)
        t_global = time.time() - t0
        result['Pooled'] = get_metrics_aft(theta_global, theta_true, t_global)
        
    if params.get('run_dgd', True):
        t0 = time.time()
        theta_dgd = run_dgd(d_aft, T=params['T'] * params['W_inner'], lr=0.1)
        t_dgd = time.time() - t0
        result['D-subGD'] = get_metrics_aft(theta_dgd, theta_true, t_dgd)
        
    return result
