import numpy as np
from utils.math_utils import _proj_sphere, soft_threshold
from models.ranking import rank_grad, rank_loss, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_pairs
from algorithms.admm import local_gd, compute_ic

def _solve_local_node(gfn, lfn, init, lam, task, p, n_iter=500, tol=1e-5):
    """
    单节点求解器：使用与 Global 完全相同的优化引擎（L-BFGS-B / SLSQP + 变量拆分），
    在单个节点的本地数据上求解带 L1 惩罚的 U-统计量 ERM。
    """
    from scipy.optimize import minimize

    if task == 'ranking':
        # 改用近端梯度下降(PGD)处理非凸球面约束
        theta_opt = local_gd(gfn, lfn, init, n_iter=n_iter, lr_init=1.0, project=True, lam=lam, decay_rate=1.0)
    else:
        if lam > 0:
            # L1 正则化：变量拆分 theta = u - v, u>=0, v>=0
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

            u0 = np.maximum(init.flatten(), 0)
            v0 = np.maximum(-init.flatten(), 0)
            x0 = np.concatenate([u0, v0])
            bounds = [(0, None)] * (2 * p)

            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': n_iter, 'gtol': tol, 'disp': False})
            theta_opt = (res.x[:p] - res.x[p:]).reshape(-1, 1)
        else:
            def obj_grad(x):
                theta = x.reshape(-1, 1)
                return float(lfn(theta)), gfn(theta).flatten()

            x0 = init.flatten()
            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': n_iter, 'gtol': tol, 'disp': False})
            theta_opt = res.x.reshape(-1, 1)

    # 清理浮点底噪 + 投影
    if task == 'ranking':
        theta_opt = _proj_sphere(theta_opt)
    theta_opt[np.abs(theta_opt) < 1e-5] = 0.0
    if task == 'ranking':
        theta_opt = _proj_sphere(theta_opt)
    return theta_opt


def run_local_penalized(data, lambda_candidates=None, ic_type='bic', init_theta_list=None, n_iter=500):
    """
    Penalized Local Estimator（带惩罚的局部估计量）。
    每个节点独立求解带 L1 惩罚的 U-统计量 ERM，使用与 Global 完全相同的
    单节点求解引擎 (_solve_local_node)。
    优化引擎（L-BFGS-B / SLSQP + 变量拆分），区别仅在于数据范围为本地 n 个样本。
    BIC 基于本地样本量 n 自动选择最优 lambda。

    Returns
    -------
    theta_list : list of (p,1) arrays, 每个节点的带惩罚局部估计
    theta_avg  : (p,1) array, 所有节点的带惩罚估计的简单平均 (即 Avg 基线)
    """
    task = data['task']
    m = data['m']
    p = data['p']

    # 确保 precomputed_pairs 已存在
    if 'precomputed_pairs' not in data:
        data['precomputed_pairs'] = []
        for j in range(m):
            if task == 'ranking':
                data['precomputed_pairs'].append(ranking_pairs(data['X'][j], data['Y'][j]))
            else:
                data['precomputed_pairs'].append(
                    aft_pairs(data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma']))

    theta_list = []

    for j in range(m):
        # 构建本地损失函数和梯度函数
        if task == 'ranking':
            dX, S = data['precomputed_pairs'][j]
            gfn = lambda th, dX=dX, S=S: rank_grad(th, dX, S)
            lfn = lambda th, dX=dX, S=S: rank_loss(th, dX, S)
        else:
            dX, dlogTt, r2, r, di, dj, n_val = data['precomputed_pairs'][j]
            gfn = lambda th, _dX=dX, _dlogTt=dlogTt, _r2=r2, _r=r, _di=di, _dj=dj, _n=n_val: \
                aft_grad(th, _dX, _dlogTt, _r2, _r, _di, _dj, _n)
            lfn = lambda th, _dX=dX, _dlogTt=dlogTt, _r2=r2, _r=r, _di=di, _dj=dj, _n=n_val: \
                aft_loss(th, _dX, _dlogTt, _r2, _r, _di, _dj, _n)

        # 初始点
        if init_theta_list is not None:
            init = init_theta_list[j].copy()
        elif task == 'ranking':
            init = np.ones((p, 1)) / np.sqrt(p)
        else:
            init = np.zeros((p, 1))

        n_local = data['X'][j].shape[0]

        # Lambda 选择：BIC 基于本地样本量 n
        if lambda_candidates is not None and len(lambda_candidates) > 0:
            best_ic = float('inf')
            best_lam = 0.0
            current_init = init.copy()

            for lam in sorted(lambda_candidates, reverse=True):
                theta_tmp = _solve_local_node(gfn, lfn, current_init, lam, task, p, n_iter=n_iter)
                current_init = theta_tmp.copy()  # 连续热启动

                loss_val = lfn(theta_tmp)
                df = np.sum(np.abs(theta_tmp) > 1e-4)
                avg_loss = max(loss_val, 1e-10)

                if ic_type.lower() == 'aic':
                    ic_val = np.log(avg_loss) + (2.0 / n_local) * df
                else:
                    ic_val = np.log(avg_loss) + (np.log(n_local) / n_local) * df

                if ic_val < best_ic:
                    best_ic = ic_val
                    best_lam = lam
        else:
            best_lam = 0.0

        best_theta = _solve_local_node(gfn, lfn, init, best_lam, task, p, n_iter=n_iter)
        theta_list.append(best_theta)

    # Avg = 所有带惩罚局部估计的简单算术平均
    theta_avg = np.mean(np.hstack(theta_list), axis=1, keepdims=True)
    if task == 'ranking':
        theta_avg = _proj_sphere(theta_avg)

    return theta_list, theta_avg


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
        # 带宽 base_n 需使用节点本地样本量 n 而非总样本量 N，防止高斯核带宽过窄导致梯度消失与信息量坍缩。
        n_local = data['X'][0].shape[0]  # 每节点样本量 n
        dX, dlogTt, r2, r, di, dj_idx, n_val = aft_pairs(X_all, logTt_all, delta_all, Sigma, base_n=n_local)
        
        gfn = lambda th: aft_grad(th, dX, dlogTt, r2, r, di, dj_idx, n_val)
        lfn = lambda th: aft_loss(th, dX, dlogTt, r2, r, di, dj_idx, n_val)

    if task == 'ranking':
        init = init_theta.copy() if init_theta is not None else np.ones((p, 1)) / np.sqrt(p)
    else:
        init = init_theta.copy() if init_theta is not None else np.zeros((p, 1))
        
    def _solve_for_lambda(lam, current_init, record_history=False, max_iter=None):
        iters = max_iter if max_iter is not None else n_iter
        history_rmse = []
        if record_history and theta_true is not None:
            init_for_record = _proj_sphere(current_init) if task == 'ranking' else current_init
            history_rmse.append(float(np.linalg.norm(init_for_record - theta_true)))

        if task == 'ranking':
            # 使用近端梯度下降 (PGD) 求解非凸球面约束
            result = local_gd(gfn, lfn, current_init, n_iter=iters, lr_init=1.0, project=True, lam=lam, theta_true=theta_true if record_history else None, project_end=True, decay_rate=1.0)
            if record_history and theta_true is not None:
                theta_opt, hist_dict = result
                history_rmse.extend(hist_dict['rmse'][1:])
            else:
                theta_opt = result
        else:
            def callback(xk):
                if record_history and theta_true is not None:
                    if lam > 0:
                        th_k = (xk[:p] - xk[p:]).reshape(-1, 1)
                    else:
                        th_k = xk.reshape(-1, 1)
                    
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
                
                res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, 
                               options={'maxiter': n_iter, 'gtol': tol, 'disp': False}, callback=callback)
                    
                theta_opt = (res.x[:p] - res.x[p:]).reshape(-1, 1)
                
            else:
                def obj_grad(x):
                    theta = x.reshape(-1, 1)
                    return float(lfn(theta)), gfn(theta).flatten()
                    
                x0 = current_init.flatten()
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
                    # 若提前收敛，则用最后一次的值进行填充补齐
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
        
        # 调参阶段使用较少迭代次数加速（正式跑阶段二时才用完整 n_iter）
        tune_iters = max(n_iter // 4, 50)
        for lam in sorted_lambdas:
            # 阶段一：不记录历史，快速调参
            theta_tmp, _ = _solve_for_lambda(lam, current_init_theta, record_history=False, max_iter=tune_iters)
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
    改进版：
    1. 取消激进的步长衰减，保证后期动力与软阈值惩罚力度。
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
                # 改用恒定步长以确保参数跳出局部极小值并维持软阈值惩罚的有效性。
                lr_t = lr 
                
                theta_new = []
                for j in range(m):
                    # 1. 网络共识步骤
                    th_j = np.zeros((p, 1))
                    for k in range(m):
                        if W[j, k] > 0:
                            th_j += W[j, k] * theta[k]
                    
                    # 2. 本地梯度更新
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
                        # 强制清除浮点底噪，防止由于投影除法放大的微小数值导致后续 BIC 自由度计算错误。
                        th_j[np.abs(th_j) < 1e-5] = 0.0
                        
                    theta_new.append(th_j)
                    
                theta = theta_new
                # 移除基于容差的提前终止条件，要求算法达到指定的最大迭代次数，以保证网络充分混合与共识。
                
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
        # 此处同样保持恒定步长
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
    改进版：引入指数衰减机制，
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
                # 应用指数衰减策略，防止迭代后期发生震荡或反弹。
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
        # 在正式运行阶段应用指数衰减策略。
        current_lr = lr * (decay_rate ** t)
        theta = _step(theta, best_lam, current_lr)
        
        if return_history and theta_true is not None:
            hist_final['rmse'].append(
                float(np.mean([np.linalg.norm(theta[j] - theta_true) for j in range(m)])))

    if return_history:
        return np.mean(theta, axis=0), hist_final
    return np.mean(theta, axis=0)