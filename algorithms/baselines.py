import numpy as np
from utils.math_utils import _proj_sphere
from models.ranking import rank_grad, rank_loss, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_pairs
from algorithms.admm import local_gd

def run_global_u_erm(data, lr=0.5, n_iter=300):
    """
    Pooled MR (Global U-ERM): 将所有本地数据汇总到一台机器上，利用全部数据计算估计值。
    """
    task = data['task']
    p = data['p']
    
    if task == 'ranking':
        X_all = np.vstack(data['X'])
        Y_all = np.concatenate(data['Y'])
        dX, S = ranking_pairs(X_all, Y_all)
        
        gfn = lambda th: rank_grad(th, dX, S)
        lfn = lambda th: rank_loss(th, dX, S)
        init = np.ones((p, 1)) / np.sqrt(p)
        
        theta_global = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=True)
        return theta_global
        
    elif task == 'aft':
        X_all = np.vstack(data['X'])
        logTt_all = np.concatenate(data['logTt'])
        delta_all = np.concatenate(data['delta'])
        Sigma = data['Sigma']
        
        dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(X_all, logTt_all, delta_all, Sigma)
        
        gfn = lambda th: aft_grad(th, dX, dlogTt, r2, r, di, dj, n_val)
        lfn = lambda th: aft_loss(th, dX, dlogTt, r2, r, di, dj, n_val)
        init = np.zeros((p, 1))
        
        theta_global = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=False)
        return theta_global

def run_dgd(data, T=50, lr=0.1):
    """
    D-subGD (Decentralized Gradient Descent): 节点通过本地梯度下降和网络通信协作求解。
    """
    m = data['m']
    p = data['p']
    W = data['W']
    task = data['task']
    
    # 预计算本地 pairs 以加速
    local_pairs = []
    for j in range(m):
        if task == 'ranking':
            dX, S = ranking_pairs(data['X'][j], data['Y'][j])
            local_pairs.append((dX, S))
        elif task == 'aft':
            dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma'])
            local_pairs.append((dX, dlogTt, r2, r, di, dj, n_val))
            
    # 初始化
    if task == 'ranking':
        theta = [np.ones((p, 1)) / np.sqrt(p) for _ in range(m)]
    else:
        theta = [np.zeros((p, 1)) for _ in range(m)]
        
    # 迭代
    for t in range(T):
        theta_new = []
        for j in range(m):
            # 1. Consensus step (网络通信)
            th_j = np.zeros((p, 1))
            for k in range(m):
                if W[j, k] > 0:
                    th_j += W[j, k] * theta[k]
            
            # 2. Local Gradient step
            if task == 'ranking':
                dX, S = local_pairs[j]
                g = rank_grad(th_j, dX, S)
            else:
                dX, dlogTt, r2, r, di, dj, n_val = local_pairs[j]
                g = aft_grad(th_j, dX, dlogTt, r2, r, di, dj, n_val)
                
            th_j = th_j - lr * g
            
            if task == 'ranking':
                th_j = _proj_sphere(th_j)
                
            theta_new.append(th_j)
        theta = theta_new
        
    # 返回平均值作为最终估计
    return np.mean(theta, axis=0)
