import os
# 🟢 终极并行加速：必须在 import numpy 之前锁死底层多线程！
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import time
import numpy as np
from models.ranking import generate_ranking_data
from models.aft import generate_aft_data
from algorithms.admm import run_u_admm, init_all_nodes
from algorithms.baselines import run_global_u_erm, run_dgd
from utils.eval_utils import evaluate_ranking_accuracy, calculate_metrics, evaluate_correlation

def get_metrics_ranking(theta, theta_true, X, Y, quantiles, t_cost):
    metrics = calculate_metrics(theta_true, theta)
    acc_dict = evaluate_ranking_accuracy(X, Y, theta, quantiles)
    corr_dict = evaluate_correlation(X, theta_true, theta)
    
    return {
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'Selection_Acc': metrics['Selection_Acc'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1_Score': metrics['F1_Score'],
        'Pairwise_Correlation': float(acc_dict['Pairwise_Correlation']),
        'Pearson_Corr': corr_dict['Pearson_Corr'],
        'Kendall_Corr': corr_dict['Kendall_Corr'],
        'Total_Pairs': acc_dict['Total_Pairs'],
        'Correct_Pairs': acc_dict['Correct_Pairs'],
        'Time': float(t_cost)
    }

def get_metrics_aft(theta, theta_true, X, t_cost):
    metrics = calculate_metrics(theta_true, theta)
    corr_dict = evaluate_correlation(X, theta_true, theta)
    
    return {
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'Selection_Acc': metrics['Selection_Acc'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1_Score': metrics['F1_Score'],
        'Pairwise_Correlation': (corr_dict['Kendall_Corr'] + 1) / 2.0,
        'Pearson_Corr': corr_dict['Pearson_Corr'],
        'Kendall_Corr': corr_dict['Kendall_Corr'],
        'Time': float(t_cost)
    }

def run_single_ranking(seed, params):
    np.random.seed(seed)
    d_rank = generate_ranking_data(
        m=params['m'], n=params['n'], p_prime=params['p_prime'], 
        p=params['p'], pc=params['pc'], noise_type=params['noise_type'], rng_seed=seed,noise_scale=params.get('noise_scale', 1.0)
    )
    
    theta_true = d_rank['theta_true']
    X, Y, quantiles = d_rank['X'], d_rank['Y'], d_rank['quantiles']
    
    # 【核心修复】统一执行一次热启动初始化，所有算法共享起点
    theta0_list, theta_naive = init_all_nodes(d_rank)
    
    result = {'seed': seed, 'noise_type': params['noise_type']}
    
    lambda_candidates = params.get('lambda_candidates', [0.1, 0.05, 0.01, 0.005, 0.001])
    ic_type = params.get('ic_type', 'bic')

    if params.get('run_proposed', True):
        t0 = time.time()
        theta_u_r, theta_n_r, hist_r = run_u_admm(
            d_rank, T=params['T'], W_inner=params['W_inner'], 
            rho=params['rho'], verbose=False,
            lambda_candidates=lambda_candidates,
            ic_type=ic_type,
            theta0_list=theta0_list  # 传递热启动
        )
        t_uadmm = time.time() - t0
        result['Proposed'] = get_metrics_ranking(theta_u_r[0], theta_true, X, Y, quantiles, t_uadmm)
        result['Proposed']['hist_rmse'] = hist_r['rmse']
        result['Proposed']['theta_hat'] = theta_u_r[0].flatten().tolist()
        
        # Avg MR (Naive) 直接取自热启动平均值，已包含在 init_all_nodes 的 theta_naive 中
        result['Avg'] = get_metrics_ranking(theta_naive, theta_true, X, Y, quantiles, 0.0)
        
    if params.get('run_local', True):
        t0 = time.time()
        # 直接使用预先算好的 theta0_list
        t_local = time.time() - t0
        local_rmses, local_maes, local_accs = [], [], []
        for th in theta0_list:
            m_dict = get_metrics_ranking(th, theta_true, X, Y, quantiles, 0)
            local_rmses.append(m_dict['RMSE'])
            local_maes.append(m_dict['MAE'])
            local_accs.append(m_dict['Pairwise_Correlation'])
        result['Local'] = {
            'RMSE': float(np.mean(local_rmses)),
            'MAE': float(np.mean(local_maes)),
            'Pairwise_Correlation': float(np.mean(local_accs)),
            'Time': float(t_local)
        }
        
    if params.get('run_baselines', True):
        t0 = time.time()
        total_iters = params['T'] * params['W_inner']
        # 传递 theta_naive 作为 Pooled 的初始化点
        theta_global, hist_global = run_global_u_erm(d_rank, n_iter=total_iters, lambda_candidates=lambda_candidates, ic_type=ic_type, init_theta=theta_naive, return_history=True)
        t_global = time.time() - t0
        result['Pooled'] = get_metrics_ranking(theta_global, theta_true, X, Y, quantiles, t_global)
        result['Pooled']['hist_rmse'] = hist_global['rmse']
        result['Pooled']['theta_hat'] = theta_global.flatten().tolist()
        
    if params.get('run_baselines', True):
        t0 = time.time()
        # 传递 theta0_list 作为 D-subGD 的初始分布
        theta_dgd, hist_dgd = run_dgd(d_rank, T=params['T'] * params['W_inner'], lr=0.1, lambda_candidates=lambda_candidates, ic_type=ic_type, theta_init_list=theta0_list, return_history=True)
        t_dgd = time.time() - t0
        result['D-subGD'] = get_metrics_ranking(theta_dgd, theta_true, X, Y, quantiles, t_dgd)
        result['D-subGD']['hist_rmse'] = hist_dgd['rmse']
        result['D-subGD']['theta_hat'] = theta_dgd.flatten().tolist()
        
    return result

def run_single_aft(seed, params):
    np.random.seed(seed)
    d_aft = generate_aft_data(
        m=params['m'], n=params['n'], p_prime=params.get('p_prime', 5), p=params['p'], 
        pc=params['pc'], noise_type=params['noise_type'], rng_seed=seed,noise_scale=params.get('noise_scale', 1.0)
    )
    
    theta_true = d_aft['theta_true']
    
    # 统一热启动
    theta0_list, theta_naive = init_all_nodes(d_aft)
    
    result = {'seed': seed, 'noise_type': params['noise_type']}
    
    lambda_candidates = params.get('lambda_candidates', [0.1, 0.05, 0.01, 0.005, 0.001])
    ic_type = params.get('ic_type', 'bic')

    if params.get('run_proposed', True):
        t0 = time.time()
        theta_u_a, theta_n_a, hist_a = run_u_admm(
            d_aft, T=params['T'], W_inner=params['W_inner'], 
            rho=params['rho'], verbose=False,
            lambda_candidates=lambda_candidates,
            ic_type=ic_type,
            theta0_list=theta0_list
        )
        t_uadmm = time.time() - t0
        result['Proposed'] = get_metrics_aft(theta_u_a[0], theta_true, d_aft['X'], t_uadmm)
        result['Proposed']['hist_rmse'] = hist_a['rmse']
        result['Proposed']['theta_hat'] = theta_u_a[0].flatten().tolist()
        
        result['Avg'] = get_metrics_aft(theta_naive, theta_true, d_aft['X'], 0.0)
        
    if params.get('run_local', True):
        t0 = time.time()
        t_local = time.time() - t0
        local_rmses, local_maes = [], []
        for th in theta0_list:
            m_dict = get_metrics_aft(th, theta_true, d_aft['X'], 0)
            local_rmses.append(m_dict['RMSE'])
            local_maes.append(m_dict['MAE'])
        result['Local'] = {
            'RMSE': float(np.mean(local_rmses)),
            'MAE': float(np.mean(local_maes)),
            'Time': float(t_local)
        }
        
    if params.get('run_baselines', True):
        t0 = time.time()
        total_iters = params['T'] * params['W_inner']
        theta_global, hist_global = run_global_u_erm(d_aft, n_iter=total_iters, lambda_candidates=lambda_candidates, ic_type=ic_type, init_theta=theta_naive, return_history=True)
        t_global = time.time() - t0
        result['Pooled'] = get_metrics_aft(theta_global, theta_true, d_aft['X'], t_global)
        result['Pooled']['hist_rmse'] = hist_global['rmse']
        result['Pooled']['theta_hat'] = theta_global.flatten().tolist()
        
    if params.get('run_baselines', True):
        t0 = time.time()
        theta_dgd, hist_dgd = run_dgd(d_aft, T=params['T'] * params['W_inner'], lr=0.1, lambda_candidates=lambda_candidates, ic_type=ic_type, theta_init_list=theta0_list, return_history=True)
        t_dgd = time.time() - t0
        result['D-subGD'] = get_metrics_aft(theta_dgd, theta_true, d_aft['X'], t_dgd)
        result['D-subGD']['hist_rmse'] = hist_dgd['rmse']
        result['D-subGD']['theta_hat'] = theta_dgd.flatten().tolist()
        
    return result
