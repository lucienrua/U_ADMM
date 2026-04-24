import numpy as np
from utils.math_utils import _proj_sphere, soft_threshold
from models.ranking import rank_grad, rank_loss, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_pairs
from algorithms.admm import local_gd, compute_ic

# def run_global_u_erm(data, lr=0.5, n_iter=300, lambda_candidates=None, ic_type='bic', init_theta=None, return_history=False, tol=1e-5):
#     """
#     Pool (Global U-ERM): 将所有本地数据汇总到一台机器上。
#     引入正则化路径 (降序排列) + 连续热启动 + 双阶段解耦。
#     """
#     task = data['task']
#     p = data['p']
#     theta_true = data.get('theta_true', None) if return_history else None
    
#     if task == 'ranking':
#         X_all = np.vstack(data['X'])
#         Y_all = np.concatenate(data['Y'])
#         dX, S = ranking_pairs(X_all, Y_all)
        
#         gfn = lambda th: rank_grad(th, dX, S)
#         lfn = lambda th: rank_loss(th, dX, S)
#         init = init_theta.copy() if init_theta is not None else np.ones((p, 1)) / np.sqrt(p)
#         project = True
        
#     elif task == 'aft':
#         X_all = np.vstack(data['X'])
#         logTt_all = np.concatenate(data['logTt'])
#         delta_all = np.concatenate(data['delta'])
#         Sigma = data['Sigma']
        
#         dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(X_all, logTt_all, delta_all, Sigma)
        
#         gfn = lambda th: aft_grad(th, dX, dlogTt, r2, r, di, dj, n_val)
#         lfn = lambda th: aft_loss(th, dX, dlogTt, r2, r, di, dj, n_val)
#         init = init_theta.copy() if init_theta is not None else np.zeros((p, 1))
#         project = False
        
#     if lambda_candidates is not None and len(lambda_candidates) > 0:
#         best_ic = float('inf')
#         best_lam = 0.0
#         N_total = sum(data['X'][j].shape[0] for j in range(data['m']))
        
#         # 核心改造 1：强制降序排列 lambda_candidates
#         sorted_lambdas = sorted(lambda_candidates, reverse=True)
#         # 核心改造 2：建立流动的热启动起点
#         current_init_theta = init.copy()
        
#         # --- 阶段一：极速调参寻找最优 lambda ---
#         for lam in sorted_lambdas:
#             # 开启提前终止 (如果 local_gd 支持 tol) 进行极速收敛，不记录 history
#             theta_tmp = local_gd(gfn, lfn, current_init_theta, n_iter=n_iter, lr_init=lr, project=project, lam=lam)
            
#             # 核心改造 3：用当前收敛的参数更新流动起点，喂给下一个更小的 lam
#             current_init_theta = theta_tmp.copy()
            
#             loss_val = lfn(theta_tmp)
#             df = np.sum(np.abs(theta_tmp) > 1e-4)
#             avg_loss = loss_val if loss_val > 0 else 1e-10
            
#             if ic_type.lower() == 'aic':
#                 ic_val = np.log(avg_loss) + (2.0 / N_total) * df
#             else:
#                 ic_val = np.log(avg_loss) + (np.log(N_total) / N_total) * df
            
#             if ic_val < best_ic:
#                 best_ic = ic_val
#                 best_lam = lam
#     else:
#         best_lam = 0.0

#     # --- 阶段二：画图阶段 (使用全局最优 lambda 严格跑满 n_iter 轮) ---
#     if return_history:
#         # 回退到原始起点 init，严格跑满并记录 history
#         best_theta, best_history = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=best_lam, theta_true=theta_true)
#         return best_theta, best_history
#     else:
#         best_theta = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=lr, project=project, lam=best_lam)
#         return best_theta
# #以上的方法是继承制lam，以下的算法是并行lam
def run_global_u_erm(data, lr=0.5, n_iter=300, lambda_candidates=None, ic_type='bic', init_theta=None, return_history=False, tol=1e-5):
    """
    Pool (Global U-ERM): 将所有本地数据汇总到一台机器上。
    取消热启动，每个 lambda 独立从头开始，保证绝对公平的评估。
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
        
        sorted_lambdas = sorted(lambda_candidates, reverse=True)
        
        # --- 阶段一：极速调参寻找最优 lambda ---
        for lam in sorted_lambdas:
            # 🔴 核心修复：绝对公平原则！取消热启动，每次测试新的 lam 都从最原始的 init.copy() 重新出发
            theta_tmp = local_gd(gfn, lfn, init.copy(), n_iter=n_iter, lr_init=lr, project=project, lam=lam)
            
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
        best_theta, best_history = local_gd(gfn, lfn, init.copy(), n_iter=n_iter, lr_init=lr, project=project, lam=best_lam, theta_true=theta_true)
        return best_theta, best_history
    else:
        best_theta = local_gd(gfn, lfn, init.copy(), n_iter=n_iter, lr_init=lr, project=project, lam=best_lam)
        return best_theta

def run_dgd(data, T=500, lr=0.1, lambda_candidates=None, ic_type='bic', theta_init_list=None, return_history=False):
    """
    标准的 D-subGD (Decentralized Subgradient Descent)
    严格遵循理论设计：
    1. 采用标准的 1/sqrt(t) 步长衰减。
    2. 无底层数值硬截断，保留纯粹的近端映射与投影结果。
    3. 运行完整的 T 轮迭代，不设基于变量差值的提前终止。
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
            from models.ranking import ranking_pairs, rank_grad
            dX, S = ranking_pairs(data['X'][j], data['Y'][j])
            local_pairs.append((dX, S))
        elif task == 'aft':
            from models.aft import aft_pairs, aft_grad
            dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma'])
            local_pairs.append((dX, dlogTt, r2, r, di, dj, n_val))
            
    # 初始化
    if theta_init_list is not None:
        init_theta = [th.copy() for th in theta_init_list]
    elif task == 'ranking':
        init_theta = [np.ones((p, 1)) / np.sqrt(p) for _ in range(m)]
    else:
        init_theta = [np.zeros((p, 1)) for _ in range(m)]
        
    if lambda_candidates is not None and len(lambda_candidates) > 0:
        best_ic = float('inf')
        best_lam = 0.0
        sorted_lambdas = sorted(lambda_candidates, reverse=True)
        
        # --- 阶段一：调参寻找最优 lambda ---
        for lam_cand in sorted_lambdas:
            theta = [th.copy() for th in init_theta]
            
            for t in range(1, T + 1):
                # 标准理论衰减步长
                lr_t = lr / np.sqrt(t) 
                
                theta_new = []
                for j in range(m):
                    # 1. Consensus step (网络共识)
                    th_j = np.zeros((p, 1))
                    for k in range(m):
                        if W[j, k] > 0:
                            th_j += W[j, k] * theta[k]
                    
                    # 2. Local Gradient step (本地次梯度)
                    if task == 'ranking':
                        dX, S = local_pairs[j]
                        g = rank_grad(th_j, dX, S)
                    else:
                        dX, dlogTt, r2, r, di, dj, n_val = local_pairs[j]
                        g = aft_grad(th_j, dX, dlogTt, r2, r, di, dj, n_val)
                        
                    th_j = th_j - lr_t * g
                    
                    # 3. Proximal step (软阈值近端映射)
                    if lam_cand > 0:
                        th_j = soft_threshold(th_j, lr_t * lam_cand)
                    
                    # 4. Projection (变量空间投影)
                    if task == 'ranking':
                        th_j = _proj_sphere(th_j)
                        
                    theta_new.append(th_j)
                    
                theta = theta_new
                
            ic_val = compute_ic(theta, data, ic_type=ic_type)
            if ic_val < best_ic:
                best_ic = ic_val
                best_lam = lam_cand
    else:
        best_lam = 0.0

    # --- 阶段二：使用全局最优 lambda 严格跑满 T 轮 ---
    theta = [th.copy() for th in init_theta]  
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

# def run_dgd(data, T=500, lr=0.1, lambda_candidates=None, ic_type='bic', theta_init_list=None, return_history=False, tol=1e-4):
#     """
#     D-subGD (Decentralized Subgradient Descent)
#     【修复版】：
#     1. 取消激进的 1/sqrt(t) 步长衰减，保证后期动力与软阈值惩罚力度。
#     2. 禁止提前终止，强制网络充分混合。
#     3. 增加投影后的浮点去噪，保证 BIC 自由度 (df) 评估的真实性。
#     """
#     m = data['m']
#     p = data['p']
#     W = data['W']
#     task = data['task']
#     theta_true = data.get('theta_true', None) if return_history else None
    
#     # 预计算本地 pairs 以加速
#     local_pairs = []
#     for j in range(m):
#         if task == 'ranking':
#             from models.ranking import ranking_pairs, rank_grad
#             dX, S = ranking_pairs(data['X'][j], data['Y'][j])
#             local_pairs.append((dX, S))
#         elif task == 'aft':
#             from models.aft import aft_pairs, aft_grad
#             dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma'])
#             local_pairs.append((dX, dlogTt, r2, r, di, dj, n_val))
            
#     # 初始化：确保所有算法起跑线一致 (独立冷启动)
#     if theta_init_list is not None:
#         init_theta = [th.copy() for th in theta_init_list]
#     elif task == 'ranking':
#         init_theta = [np.ones((p, 1)) / np.sqrt(p) for _ in range(m)]
#     else:
#         init_theta = [np.zeros((p, 1)) for _ in range(m)]
        
#     if lambda_candidates is not None and len(lambda_candidates) > 0:
#         from algorithms.admm import compute_ic
#         from utils.math_utils import soft_threshold, _proj_sphere
        
#         best_ic = float('inf')
#         best_lam = 0.0
        
#         # 降序排列候选列表
#         sorted_lambdas = sorted(lambda_candidates, reverse=True)
        
#         # --- 阶段一：调参寻找最优 lambda ---
#         for lam_cand in sorted_lambdas:
#             # 绝对隔离原则：每次测试新的 lam 都从最原始的起点重新出发
#             theta = [th.copy() for th in init_theta]
            
#             for t in range(1, T + 1):
#                 # 🔴 修复 1：取消 1/sqrt(t) 衰减，采用恒定步长 (或极其缓慢的衰减)
#                 # 这保证了参数能走出局部区域，且 lr_t * lam_cand 始终具备压制力
#                 lr_t = lr 
                
#                 theta_new = []
#                 for j in range(m):
#                     # 1. Consensus step (网络共识)
#                     th_j = np.zeros((p, 1))
#                     for k in range(m):
#                         if W[j, k] > 0:
#                             th_j += W[j, k] * theta[k]
                    
#                     # 2. Local Gradient step (本地梯度)
#                     if task == 'ranking':
#                         dX, S = local_pairs[j]
#                         g = rank_grad(th_j, dX, S)
#                     else:
#                         dX, dlogTt, r2, r, di, dj, n_val = local_pairs[j]
#                         g = aft_grad(th_j, dX, dlogTt, r2, r, di, dj, n_val)
                        
#                     # 3. 次梯度下降
#                     th_j = th_j - lr_t * g
                    
#                     # 4. 近端算子 (软阈值)
#                     if lam_cand > 0:
#                         th_j = soft_threshold(th_j, lr_t * lam_cand)
                    
#                     # 5. 变量空间投影
#                     if task == 'ranking':
#                         th_j = _proj_sphere(th_j)
#                         # 🔴 修复 4：由于 _proj_sphere 的除法操作会放大微小数值
#                         # 必须强制将浮点底噪归零，否则后续 BIC 计算的 df 永远是满秩 p
#                         th_j[np.abs(th_j) < 1e-5] = 0.0
                        
#                     theta_new.append(th_j)
                    
#                 theta = theta_new
#                 # 🔴 修复 3：彻底删除 max_diff < tol 的提前终止逻辑。
#                 # 强制算法跑满 T 轮，避免在尚未达成网络共识时虚假收敛。
                
#             # 计算信息准则
#             ic_val = compute_ic(theta, data, ic_type=ic_type)
#             if ic_val < best_ic:
#                 best_ic = ic_val
#                 best_lam = lam_cand
#     else:
#         best_lam = 0.0
#         from utils.math_utils import soft_threshold, _proj_sphere

#     # --- 阶段二：使用最优 lambda 严格跑满 T 轮记录历史 ---
#     theta = [th.copy() for th in init_theta]  
#     hist_final = {'rmse': []}
    
#     if return_history and theta_true is not None:
#         rmse = float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)]))
#         hist_final['rmse'].append(rmse)
        
#     for t in range(1, T + 1):
#         # 🔴 这里同样保持恒定步长
#         lr_t = lr 
        
#         theta_new = []
#         for j in range(m):
#             th_j = np.zeros((p, 1))
#             for k in range(m):
#                 if W[j, k] > 0:
#                     th_j += W[j, k] * theta[k]
            
#             if task == 'ranking':
#                 dX, S = local_pairs[j]
#                 g = rank_grad(th_j, dX, S)
#             else:
#                 dX, dlogTt, r2, r, di, dj, n_val = local_pairs[j]
#                 g = aft_grad(th_j, dX, dlogTt, r2, r, di, dj, n_val)
                
#             th_j = th_j - lr_t * g
            
#             if best_lam > 0:
#                 th_j = soft_threshold(th_j, lr_t * best_lam)
                
#             if task == 'ranking':
#                 th_j = _proj_sphere(th_j)
#                 th_j[np.abs(th_j) < 1e-5] = 0.0 # 保持画图阶段也是干净的稀疏结构
                
#             theta_new.append(th_j)
#         theta = theta_new
        
#         if return_history and theta_true is not None:
#             rmse = float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)]))
#             hist_final['rmse'].append(rmse)
            
#     if return_history:
#         return np.mean(theta, axis=0), hist_final
#     return np.mean(theta, axis=0)

def run_d_proxgd(data, T=500, lr=0.1, lambda_candidates=None, ic_type='bic', 
             theta_init_list=None, return_history=False, decay_interval=100, decay_rate=0.5):
    """
    D-ProxGD (Decentralized Proximal Gradient Descent)
    【修复版】：恢复了阶梯衰减机制，保证前期强力去噪，后期高精度收敛。
    """
    m = data['m']
    p = data['p']
    W = data['W']
    task = data['task']
    theta_true = data.get('theta_true', None) if return_history else None

    # ── 预计算本地 pairs ──────────────────────────────────────────────
    local_pairs = []
    for j in range(m):
        if task == 'ranking':
            from models.ranking import ranking_pairs, rank_grad
            dX, S = ranking_pairs(data['X'][j], data['Y'][j])
            local_pairs.append((dX, S))
        elif task == 'aft':
            from models.aft import aft_pairs, aft_grad
            dX, dlogTt, r2, r, di, dj_idx, n_val = aft_pairs(
                data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma'])
            local_pairs.append((dX, dlogTt, r2, r, di, dj_idx, n_val))

    # ── 初始化起点 ────────────────────────────────────────────────────
    if theta_init_list is not None:
        init_theta = [th.copy() for th in theta_init_list]
    elif task == 'ranking':
        init_theta = [np.ones((p, 1)) / np.sqrt(p) for _ in range(m)]
    else:
        init_theta = [np.zeros((p, 1)) for _ in range(m)]

    # ── 单轮迭代（带动态步长） ─────────────────────────────────────────
    def _step(theta, lam, current_lr):
        theta_new = []
        for j in range(m):
            v_j = np.zeros((p, 1))
            for k in range(m):
                if W[j, k] > 0:
                    v_j += W[j, k] * theta[k]
                    
            if task == 'ranking':
                dX, S = local_pairs[j]
                g = rank_grad(v_j, dX, S)
            else:
                dX, dlogTt, r2, r, di, dj_idx, n_val = local_pairs[j]
                g = aft_grad(v_j, dX, dlogTt, r2, r, di, dj_idx, n_val)
                
            v_j = v_j - current_lr * g
            
            if lam > 0:
                v_j = soft_threshold(v_j, current_lr * lam)
                
            if task == 'ranking':
                v_j = _proj_sphere(v_j)
                v_j[np.abs(v_j) < 1e-5] = 0.0
            theta_new.append(v_j)
        return theta_new

    # ── 阶段一：冷启动调参 ────────────────────────────────────────────
    if lambda_candidates is not None and len(lambda_candidates) > 0:
        best_ic, best_lam = float('inf'), 0.0
        for lam_cand in sorted(lambda_candidates, reverse=True):
            theta = [th.copy() for th in init_theta]
            for t in range(T):
                # 阶梯衰减逻辑
                current_lr = lr * (decay_rate ** (t // decay_interval))
                theta = _step(theta, lam_cand, current_lr)
                
            ic_val = compute_ic(theta, data, ic_type=ic_type)
            if ic_val < best_ic:
                best_ic, best_lam = ic_val, lam_cand
    else:
        best_lam = 0.0

    # ── 阶段二：最优 lambda 跑满 T 轮，记录历史 ──────────────────────
    theta = [th.copy() for th in init_theta]
    hist_final = {'rmse': []}

    if return_history and theta_true is not None:
        hist_final['rmse'].append(
            float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)])))

    for t in range(T):
        # 同样的阶梯衰减逻辑
        current_lr = lr * (decay_rate ** (t // decay_interval))
        theta = _step(theta, best_lam, current_lr)
        
        if return_history and theta_true is not None:
            hist_final['rmse'].append(
                float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)])))

    if return_history:
        return np.mean(theta, axis=0), hist_final
    return np.mean(theta, axis=0)