import numpy as np
from utils.math_utils import _proj_sphere
from models.ranking import rank_grad, rank_loss, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_pairs
from algorithms.admm import local_gd

def run_global_u_erm(data, lr=0.5, n_iter=300, lambda_candidates=None, ic_type='bic', init_theta=None, return_history=False):
    """
    Pooled MR (Global U-ERM): 将所有本地数据汇总到一台机器上，利用全部数据计算估计值。
    init_theta: 外部热启动参数 (p,1)，若为 None 则使用默认初始化。
    """
    task = data['task']
    p = data['p']
    theta_true = data.get('theta_true', None) if return_history else None
    
    if task == 'ranking':
        X_all = np.vstack(data['X'])
        Y_all = np.concatenate(data['Y'])
        dX, S = ranking_pairs(X_all, Y_all)
        
        gfn = lambda th: rank_grad(th, dX, S)
        lfn = lambda th: rank_loss(th, dX, S)
        # 热启动：优先使用传入的 init_theta，否则回退到归一化均匀向量
        init = init_theta.copy() if init_theta is not None else np.ones((p, 1)) / np.sqrt(p)
        project = True
        
    elif task == 'aft':
        X_all = np.vstack(data['X'])
        logTt_all = np.concatenate(data['logTt'])
        delta_all = np.concatenate(data['delta'])
        Sigma = data['Sigma']
        
        dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(X_all, logTt_all, delta_all, Sigma)
        
        gfn = lambda th: aft_grad(th, dX, dlogTt, r2, r, di, dj, n_val)
        lfn = lambda th: aft_loss(th, dX, dlogTt, r2, r, di, dj, n_val)
        init = init_theta.copy() if init_theta is not None else np.zeros((p, 1))
        project = False
        
    if lambda_candidates is not None:
        best_theta = None
        best_ic = float('inf')
        best_history = None
        N_total = sum(data['X'][j].shape[0] for j in range(data['m']))
        
        for lam in lambda_candidates:
            if return_history:
                theta_tmp, hist_tmp = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=lam, theta_true=theta_true)
            else:
                theta_tmp = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=lam)
            
            loss_val = lfn(theta_tmp)
            df = np.sum(np.abs(theta_tmp) > 1e-4)
            
            # loss_val 本身即为平均损失 (avg_loss)
            avg_loss = loss_val if loss_val > 0 else 1e-10
            
            if ic_type.lower() == 'aic':
                ic_val = np.log(avg_loss) + (2.0 / N_total) * df
            else:
                ic_val = np.log(avg_loss) + (np.log(N_total) / N_total) * df
            
            if ic_val < best_ic:
                best_ic = ic_val
                best_theta = theta_tmp
                if return_history:
                    best_history = hist_tmp
                
        if return_history:
            return best_theta, best_history
        return best_theta
    else:
        if return_history:
            return local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=0.0, theta_true=theta_true)
        return local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=0.0)

def run_dgd(data, T=50, lr=0.1, lambda_candidates=None, ic_type='bic', theta_init_list=None, return_history=False):
    """
    D-subGD (Decentralized Gradient Descent): 节点通过本地梯度下降和网络通信协作求解。
    theta_init_list: 外部热启动参数列表（与 m 等长），若为 None 则使用默认初始化。
    """
    m = data['m']
    p = data['p']
    W = data['W']
    task = data['task']
    theta_true = data.get('theta_true', None) if return_history else None
    
    # 预计算本地 pairs 以加速
    local_pairs = []
    for j in range(m):
        if task == 'ranking':
            dX, S = ranking_pairs(data['X'][j], data['Y'][j])
            local_pairs.append((dX, S))
        elif task == 'aft':
            dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma'])
            local_pairs.append((dX, dlogTt, r2, r, di, dj, n_val))
            
    # 初始化：优先使用热启动，否则使用默认初始值
    if theta_init_list is not None:
        init_theta = [th.copy() for th in theta_init_list]
    elif task == 'ranking':
        init_theta = [np.ones((p, 1)) / np.sqrt(p) for _ in range(m)]
    else:
        init_theta = [np.zeros((p, 1)) for _ in range(m)]
        
    if lambda_candidates is not None:
        from algorithms.admm import compute_ic
        from utils.math_utils import soft_threshold
        best_theta = None
        best_ic = float('inf')
        best_history = None
        
        for lam in lambda_candidates:
            theta = [th.copy() for th in init_theta]
            hist_tmp = {'rmse': []}
            if return_history and theta_true is not None:
                hist_tmp['rmse'].append(float(np.mean([np.linalg.norm(th_iter - theta_true) for th_iter in theta])))
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
                    
                    if lam > 0:
                        th_j = soft_threshold(th_j, lr * lam)
                    
                    if task == 'ranking':
                        th_j = _proj_sphere(th_j)
                        
                    theta_new.append(th_j)
                theta = theta_new
                if return_history and theta_true is not None:
                    hist_tmp['rmse'].append(float(np.mean([np.linalg.norm(th_iter - theta_true) for th_iter in theta])))
                
            ic_val = compute_ic(theta, data, ic_type=ic_type)
            if ic_val < best_ic:
                best_ic = ic_val
                best_theta = theta
                if return_history:
                    best_history = hist_tmp
                
        if return_history:
            return np.mean(best_theta, axis=0), best_history
        return np.mean(best_theta, axis=0)
    else:
        from utils.math_utils import soft_threshold
        theta = [th.copy() for th in init_theta]
        hist_tmp = {'rmse': []}
        if return_history and theta_true is not None:
            hist_tmp['rmse'].append(float(np.mean([np.linalg.norm(th_iter - theta_true) for th_iter in theta])))
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
            if return_history and theta_true is not None:
                hist_tmp['rmse'].append(float(np.mean([np.linalg.norm(th_iter - theta_true) for th_iter in theta])))
            
        if return_history:
            return np.mean(theta, axis=0), hist_tmp
        return np.mean(theta, axis=0)
