import numpy as np
from utils.math_utils import _proj_sphere
from models.ranking import rank_grad, rank_loss, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_pairs
from algorithms.admm import local_gd

def run_global_u_erm(data, lr=0.5, n_iter=300, lambda_candidates=None, ic_type='bic', init_theta=None, return_history=False, tol=1e-5):
    """
    Pooled MR (Global U-ERM): 将所有本地数据汇总到一台机器上。
    引入正则化路径 (降序排列) + 连续热启动 + 双阶段解耦。
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
        
    if lambda_candidates is not None and len(lambda_candidates) > 0:
        best_ic = float('inf')
        best_lam = 0.0
        N_total = sum(data['X'][j].shape[0] for j in range(data['m']))
        
        # 核心改造 1：强制降序排列 lambda_candidates
        sorted_lambdas = sorted(lambda_candidates, reverse=True)
        # 核心改造 2：建立流动的热启动起点
        current_init_theta = init.copy()
        
        # --- 阶段一：极速调参寻找最优 lambda ---
        for lam in sorted_lambdas:
            # 开启提前终止 (如果 local_gd 支持 tol) 进行极速收敛，不记录 history
            theta_tmp = local_gd(gfn, lfn, current_init_theta, n_iter=n_iter, lr_init=lr, project=project, lam=lam)
            
            # 核心改造 3：用当前收敛的参数更新流动起点，喂给下一个更小的 lam
            current_init_theta = theta_tmp.copy()
            
            loss_val = lfn(theta_tmp)
            df = np.sum(np.abs(theta_tmp) > 1e-4)
            avg_loss = loss_val if loss_val > 0 else 1e-10
            
            if ic_type.lower() == 'aic':
                ic_val = np.log(avg_loss) + (2.0 / N_total) * df
            else:
                ic_val = np.log(avg_loss) + (np.log(N_total) / N_total) * df
            
            if ic_val < best_ic:
                best_ic = ic_val
                best_lam = lam
    else:
        best_lam = 0.0

    # --- 阶段二：画图阶段 (使用全局最优 lambda 严格跑满 n_iter 轮) ---
    if return_history:
        # 回退到原始起点 init，严格跑满并记录 history
        best_theta, best_history = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=best_lam, theta_true=theta_true)
        return best_theta, best_history
    else:
        best_theta = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=best_lam)
        return best_theta

def run_dgd(data, T=50, lr=0.1, lambda_candidates=None, ic_type='bic', theta_init_list=None, return_history=False, tol=1e-4):
    """
    D-subGD (Decentralized Subgradient Descent)
    采用绝对隔离的独立搜索，禁止热启动步数累积，保证 BIC 评判的绝对公平。
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
            
    # 初始化：确保所有算法起跑线一致
    if theta_init_list is not None:
        init_theta = [th.copy() for th in theta_init_list]
    elif task == 'ranking':
        init_theta = [np.ones((p, 1)) / np.sqrt(p) for _ in range(m)]
    else:
        init_theta = [np.zeros((p, 1)) for _ in range(m)]
        
    if lambda_candidates is not None and len(lambda_candidates) > 0:
        from algorithms.admm import compute_ic
        from utils.math_utils import soft_threshold
        
        best_ic = float('inf')
        best_lam = 0.0
        
        # 降序排列候选列表
        sorted_lambdas = sorted(lambda_candidates, reverse=True)
        
        # --- 阶段一：极速调参寻找最优 lambda ---
        for lam_cand in sorted_lambdas:
            # 🔴 核心修复：绝对公平原则！
            # 每次测试新的 lam，必须从最原始的起点重新出发，彻底切断热启动
            theta = [th.copy() for th in init_theta]
            
            for t in range(1, T + 1):
                lr_t = lr / np.sqrt(t) 
                theta_new = []
                max_diff = 0.0
                
                for j in range(m):
                    # 1. Consensus step
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
                        
                    # 3. 纯净的次梯度下降
                    th_j = th_j - lr_t * g
                    
                    # 4. 近端算子
                    if lam_cand > 0:
                        th_j = soft_threshold(th_j, lr_t * lam_cand)
                    
                    # 5. 变量空间投影
                    if task == 'ranking':
                        th_j = _proj_sphere(th_j)
                        
                    theta_new.append(th_j)
                    # 记录最大参数更新量
                    max_diff = max(max_diff, float(np.linalg.norm(th_j - theta[j])))
                    
                theta = theta_new
                
                # 达成共识且微小更新时，提早终止当前 lambda 的通信迭代
                if max_diff < tol:
                    break
            
            # 🔴 这里已经删除了 current_theta_init = [th.copy() for th in theta]
            # 不再记录和传递收敛状态
                
            ic_val = compute_ic(theta, data, ic_type=ic_type)
            if ic_val < best_ic:
                best_ic = ic_val
                best_lam = lam_cand
    else:
        best_lam = 0.0
        from utils.math_utils import soft_threshold

    # --- 阶段二：画图阶段 (使用最优 lambda 严格跑满 T 轮，控制变量) ---
    theta = [th.copy() for th in init_theta]  # 必须回到原点
    hist_final = {'rmse': []}
    
    if return_history and theta_true is not None:
        rmse = float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)]))
        hist_final['rmse'].append(rmse)
        
    for t in range(1, T + 1):
        lr_t = lr / np.sqrt(t)
        theta_new = []
        for j in range(m):
            th_j = np.zeros((p, 1))
            for k in range(m):
                if W[j, k] > 0:
                    th_j += W[j, k] * theta[k]
            
            if task == 'ranking':
                dX, S = local_pairs[j]
                g = rank_grad(th_j, dX, S)
            else:
                dX, dlogTt, r2, r, di, dj, n_val = local_pairs[j]
                g = aft_grad(th_j, dX, dlogTt, r2, r, di, dj, n_val)
                
            th_j = th_j - lr_t * g
            
            if best_lam > 0:
                th_j = soft_threshold(th_j, lr_t * best_lam)
                
            if task == 'ranking':
                th_j = _proj_sphere(th_j)
                
            theta_new.append(th_j)
        theta = theta_new
        
        if return_history and theta_true is not None:
            rmse = float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)]))
            hist_final['rmse'].append(rmse)
            
    if return_history:
        return np.mean(theta, axis=0), hist_final
    return np.mean(theta, axis=0)