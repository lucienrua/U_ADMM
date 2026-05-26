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
def run_global_u_erm(data, lr=0.5, n_iter=500, lambda_candidates=None, ic_type='bic', init_theta=None, return_history=False, tol=1e-5):
    """
    Pool (Global U-ERM): 中心化 Oracle 基线算法。
    使用二阶近似或拟牛顿法 (L-BFGS-B / SLSQP) 求解全局目标函数。
    对应 IEEE 2022 中 Global 的定义：全量数据集中计算。
    """
    from scipy.optimize import minimize
    
    task = data['task']
    p = data['p']
    m = data['m']
    theta_true = data.get('theta_true', None) if return_history else None
    
    if task == 'ranking':
        from models.ranking import ranking_pairs, rank_grad, rank_loss
        X_all = np.vstack(data['X'])
        Y_all = np.concatenate(data['Y'])
        dX, S = ranking_pairs(X_all, Y_all)
        
        gfn = lambda th: rank_grad(th, dX, S)
        lfn = lambda th: rank_loss(th, dX, S)
    elif task == 'aft':
        from models.aft import aft_pairs, aft_grad, aft_loss
        X_all = np.vstack(data['X'])
        logTt_all = np.concatenate(data['logTt'])
        delta_all = np.concatenate(data['delta'])
        Sigma = data['Sigma']
        
        # 全量 Pooled: 生成 C(N,2) 个全局 Pairs
        # 🔴 关键修复：带宽 base_n 必须用节点本地样本量 n（而非总样本量 N），
        #    否则高斯核带宽窄 sqrt(m) 倍，导致 Φ(z) 饱和、梯度归零、信息量坍缩。
        n_local = data['X'][0].shape[0]  # 每节点样本量 n
        dX, dlogTt, r2, r, di, dj_idx, n_val = aft_pairs(X_all, logTt_all, delta_all, Sigma, base_n=n_local)
        
        gfn = lambda th: aft_grad(th, dX, dlogTt, r2, r, di, dj_idx, n_val)
        lfn = lambda th: aft_loss(th, dX, dlogTt, r2, r, di, dj_idx, n_val)

    if task == 'ranking':
        init = init_theta.copy() if init_theta is not None else np.ones((p, 1)) / np.sqrt(p)
    else:
        init = init_theta.copy() if init_theta is not None else np.zeros((p, 1))
        
    def _solve_for_lambda(lam, current_init, record_history=False):
        history_rmse = []
        if record_history and theta_true is not None:
            history_rmse.append(float(np.linalg.norm(current_init - theta_true)))

        def callback(xk):
            if record_history and theta_true is not None:
                if lam > 0:
                    th_k = (xk[:p] - xk[p:]).reshape(-1, 1)
                else:
                    th_k = xk.reshape(-1, 1)
                
                if task == 'ranking':
                    # 即使未收敛，评估时也临时投影回球面
                    nrm = np.linalg.norm(th_k)
                    if nrm > 1e-12:
                        th_k = th_k / nrm
                history_rmse.append(float(np.linalg.norm(th_k - theta_true)))

        if lam > 0:
            # L1 正则化：使用变量拆分 theta = u - v, u>=0, v>=0
            def obj_grad(x_ext):
                u = x_ext[:p].reshape(-1, 1)
                v = x_ext[p:].reshape(-1, 1)
                theta = u - v
                loss = lfn(theta)
                grad = gfn(theta)
                
                obj = loss + lam * np.sum(u + v)
                grad_u = grad.flatten() + lam
                grad_v = -grad.flatten() + lam
                return float(obj), np.concatenate([grad_u, grad_v])
                
            u0 = np.maximum(current_init.flatten(), 0)
            v0 = np.maximum(-current_init.flatten(), 0)
            x0 = np.concatenate([u0, v0])
            bounds = [(0, None)] * (2 * p)
            
            if task == 'ranking':
                constraints = {
                    'type': 'eq',
                    'fun': lambda x: np.sum((x[:p] - x[p:])**2) - 1.0,
                    'jac': lambda x: np.concatenate([2 * (x[:p] - x[p:]), -2 * (x[:p] - x[p:])])
                }
                res = minimize(obj_grad, x0, method='SLSQP', jac=True, bounds=bounds, constraints=constraints, 
                               options={'maxiter': n_iter, 'ftol': tol, 'disp': False}, callback=callback)
            else:
                res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, 
                               options={'maxiter': n_iter, 'gtol': tol, 'disp': False}, callback=callback)
                
            theta_opt = (res.x[:p] - res.x[p:]).reshape(-1, 1)
            
        else:
            def obj_grad(x):
                theta = x.reshape(-1, 1)
                return float(lfn(theta)), gfn(theta).flatten()
                
            x0 = current_init.flatten()
            if task == 'ranking':
                constraints = {
                    'type': 'eq',
                    'fun': lambda x: np.dot(x, x) - 1.0,
                    'jac': lambda x: 2.0 * x
                }
                res = minimize(obj_grad, x0, method='SLSQP', jac=True, constraints=constraints, 
                               options={'maxiter': n_iter, 'ftol': tol, 'disp': False}, callback=callback)
            else:
                res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': n_iter, 'gtol': tol, 'disp': False}, callback=callback)
                
            theta_opt = res.x.reshape(-1, 1)

        # 严格清理浮点底噪，保证 df 评估准确，并修复投影问题
        if task == 'ranking':
            theta_opt = _proj_sphere(theta_opt)
        
        theta_opt[np.abs(theta_opt) < 1e-5] = 0.0
        
        if task == 'ranking':
            theta_opt = _proj_sphere(theta_opt)

        if record_history and theta_true is not None:
            target_len = 1 + n_iter
            if len(history_rmse) > 0:
                if len(history_rmse) < target_len:
                    # Pad with the last value if converged early
                    history_rmse.extend([history_rmse[-1]] * (target_len - len(history_rmse)))
                elif len(history_rmse) > target_len:
                    history_rmse = history_rmse[:target_len]

        return theta_opt, history_rmse

    if lambda_candidates is not None and len(lambda_candidates) > 0:
        best_ic = float('inf')
        best_lam = 0.0
        N_total = sum(data['X'][j].shape[0] for j in range(m))
        
        sorted_lambdas = sorted(lambda_candidates, reverse=True)
        current_init_theta = init.copy()
        
        for lam in sorted_lambdas:
            # 阶段一：不记录历史，快速调参
            theta_tmp, _ = _solve_for_lambda(lam, current_init_theta, record_history=False)
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

    # 阶段二：使用最佳 lambda 跑满并记录历史
    best_theta, best_history = _solve_for_lambda(best_lam, init, record_history=return_history)
    
    if return_history:
        return best_theta, {'rmse': best_history}
    else:
        return best_theta

def run_dgd(data, T=500, lr=0.1, lambda_candidates=None, ic_type='bic', theta_init_list=None, return_history=False, tol=1e-4):
    """
    D-subGD (Decentralized Subgradient Descent)
    【修复版】：
    1. 取消激进的 1/sqrt(t) 步长衰减，保证后期动力与软阈值惩罚力度。
    2. 禁止提前终止，强制网络充分混合。
    3. 增加投影后的浮点去噪，保证 BIC 自由度 (df) 评估的真实性。
    """
    m = data['m']
    p = data['p']
    W = data['W']
    task = data['task']
    theta_true = data.get('theta_true', None) if return_history else None
    
    # 预计算本地 pairs 以加速
    local_pairs = []
    N_total = data.get('N_total', m * data['n'])
    for j in range(m):
        if task == 'ranking':
            from models.ranking import ranking_pairs, rank_grad
            dX, S = ranking_pairs(data['X'][j], data['Y'][j])
            local_pairs.append((dX, S))
        elif task == 'aft':
            from models.aft import aft_pairs, aft_grad
            dX, dlogTt, r2, r, di, dj, n_val = aft_pairs(data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma'], base_n=N_total)
            local_pairs.append((dX, dlogTt, r2, r, di, dj, n_val))
            
    # 初始化：确保所有算法起跑线一致 (独立冷启动)
    if theta_init_list is not None:
        init_theta = [th.copy() for th in theta_init_list]
    elif task == 'ranking':
        init_theta = [np.ones((p, 1)) / np.sqrt(p) for _ in range(m)]
    else:
        init_theta = [np.zeros((p, 1)) for _ in range(m)]
        
    if lambda_candidates is not None and len(lambda_candidates) > 0:
        from algorithms.admm import compute_ic
        from utils.math_utils import soft_threshold, _proj_sphere
        
        best_ic = float('inf')
        best_lam = 0.0
        
        # 降序排列候选列表
        sorted_lambdas = sorted(lambda_candidates, reverse=True)
        
        # --- 阶段一：调参寻找最优 lambda ---
        for lam_cand in sorted_lambdas:
            # 绝对隔离原则：每次测试新的 lam 都从最原始的起点重新出发
            theta = [th.copy() for th in init_theta]
            
            for t in range(1, T + 1):
                # 🔴 修复 1：取消 1/sqrt(t) 衰减，采用恒定步长 (或极其缓慢的衰减)
                # 这保证了参数能走出局部区域，且 lr_t * lam_cand 始终具备压制力
                lr_t = lr 
                
                theta_new = []
                for j in range(m):
                    # 1. Consensus step (网络共识)
                    th_j = np.zeros((p, 1))
                    for k in range(m):
                        if W[j, k] > 0:
                            th_j += W[j, k] * theta[k]
                    
                    # 2. Local Gradient step (本地梯度)
                    if task == 'ranking':
                        dX, S = local_pairs[j]
                        g = rank_grad(th_j, dX, S)
                    else:
                        dX, dlogTt, r2, r, di, dj, n_val = local_pairs[j]
                        g = aft_grad(th_j, dX, dlogTt, r2, r, di, dj, n_val)
                        
                    # 3. 次梯度下降
                    th_j = th_j - lr_t * g
                    
                    # 4. 近端算子 (软阈值)
                    if lam_cand > 0:
                        th_j = soft_threshold(th_j, lr_t * lam_cand)
                    
                    # 5. 变量空间投影
                    if task == 'ranking':
                        th_j = _proj_sphere(th_j)
                        # 🔴 修复 4：由于 _proj_sphere 的除法操作会放大微小数值
                        # 必须强制将浮点底噪归零，否则后续 BIC 计算的 df 永远是满秩 p
                        th_j[np.abs(th_j) < 1e-5] = 0.0
                        
                    theta_new.append(th_j)
                    
                theta = theta_new
                # 🔴 修复 3：彻底删除 max_diff < tol 的提前终止逻辑。
                # 强制算法跑满 T 轮，避免在尚未达成网络共识时虚假收敛。
                
            # 计算信息准则
            ic_val = compute_ic(theta, data, ic_type=ic_type)
            if ic_val < best_ic:
                best_ic = ic_val
                best_lam = lam_cand
    else:
        best_lam = 0.0
        from utils.math_utils import soft_threshold, _proj_sphere

    # --- 阶段二：使用最优 lambda 严格跑满 T 轮记录历史 ---
    theta = [th.copy() for th in init_theta]  
    hist_final = {'rmse': []}
    
    if return_history and theta_true is not None:
        rmse = float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)]))
        hist_final['rmse'].append(rmse)
        
    for t in range(1, T + 1):
        # 🔴 这里同样保持恒定步长
        lr_t = lr 
        
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
                th_j[np.abs(th_j) < 1e-5] = 0.0 # 保持画图阶段也是干净的稀疏结构
                
            theta_new.append(th_j)
        theta = theta_new
        
        if return_history and theta_true is not None:
            rmse = float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)]))
            hist_final['rmse'].append(rmse)
            
    if return_history:
        return np.mean(theta, axis=0), hist_final
    return np.mean(theta, axis=0)

def run_d_proxgd(data, T=500, lr=0.1, lambda_candidates=None, ic_type='bic', 
             theta_init_list=None, return_history=False, decay_rate=0.96):
    """
    D-ProxGD (Decentralized Proximal Gradient Descent)
    【修复版】：引入指数衰减机制 (decay_rate=0.96)，
    确保算法在接近真实解时能够稳定收敛，防止步长过大导致的震荡。
    """
    m = data['m']
    p = data['p']
    W = data['W']
    task = data['task']
    theta_true = data.get('theta_true', None) if return_history else None

    # ── 预计算本地 pairs ──────────────────────────────────────────────
    local_pairs = []
    N_total = data.get('N_total', m * data['n'])
    for j in range(m):
        if task == 'ranking':
            from models.ranking import ranking_pairs, rank_grad
            dX, S = ranking_pairs(data['X'][j], data['Y'][j])
            local_pairs.append((dX, S))
        elif task == 'aft':
            from models.aft import aft_pairs, aft_grad
            dX, dlogTt, r2, r, di, dj_idx, n_val = aft_pairs(
                data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma'], base_n=N_total)
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
                # 🔴 核心修复：应用指数衰减，防止反弹
                current_lr = lr * (decay_rate ** t)
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
        # 🔴 核心修复：正式运行也应用指数衰减
        current_lr = lr * (decay_rate ** t)
        theta = _step(theta, best_lam, current_lr)
        
        if return_history and theta_true is not None:
            hist_final['rmse'].append(
                float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)])))

    if return_history:
        return np.mean(theta, axis=0), hist_final
    return np.mean(theta, axis=0)

    if return_history:
        return np.mean(theta, axis=0), hist_final
    return np.mean(theta, axis=0)