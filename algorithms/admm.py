import numpy as np
from scipy.optimize import minimize as scipy_minimize
from utils.math_utils import soft_threshold, _proj_sphere
from models.ranking import rank_grad, rank_loss, rank_hess, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_hess_diag, aft_pairs

def local_gd(grad_fn, loss_fn, init_theta, n_iter=300, lr_init=0.5, project=False, lam=0.0, theta_true=None, project_end=False):
    """
    Proximal Gradient Descent with Armijo line search.
    """
    theta = init_theta.copy()
    history = {'rmse': []}
    if theta_true is not None:
        record_th = _proj_sphere(theta) if project_end else theta
        history['rmse'].append(float(np.linalg.norm(record_th - theta_true)))
    for _ in range(n_iter):
        g = grad_fn(theta)
        l0 = loss_fn(theta)
        alpha = lr_init
        for _ in range(25):
            cand = theta - alpha * g
            if lam > 0:
                cand = soft_threshold(cand, alpha * lam)
            if project: cand = _proj_sphere(cand)

            step_diff = cand - theta
            if loss_fn(cand) <= l0 + float(np.sum(g * step_diff)) + (0.5/alpha) * np.sum(step_diff**2):
                break
            alpha *= 0.5

        cand = theta - alpha * g
        if lam > 0:
            cand = soft_threshold(cand, alpha * lam)
        if project: cand = _proj_sphere(cand)
        theta = cand
        if theta_true is not None:
            record_th = _proj_sphere(theta) if project_end else theta
            history['rmse'].append(float(np.linalg.norm(record_th - theta_true)))
            
    if project_end:
        theta = _proj_sphere(theta)
        
    if theta_true is not None:
        return theta, history
    return theta

def _slsqp_ranking_init(dX, S, p, maxiter=500):
    """
    用 SLSQP 在单位球面约束下精确最小化 Ranking Logistic 损失。
    相比一阶 local_gd 具有更快的收敛速度和更稳健的初始估计。
    """
    def obj_grad(x_flat):
        th = x_flat.reshape(-1, 1)
        return rank_loss(th, dX, S), rank_grad(th, dX, S).flatten()

    x0 = np.ones(p) / np.sqrt(p)
    sphere_constraint = {
        'type': 'eq',
        'fun': lambda x: np.dot(x, x) - 1.0,
        'jac': lambda x: 2.0 * x
    }
    result = scipy_minimize(
        obj_grad, x0, method='SLSQP', jac=True,
        constraints=sphere_constraint,
        options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False}
    )
    return _proj_sphere(result.x.reshape(-1, 1))  # 数值安全投影


def init_all_nodes(data):
    m, p = data['m'], data['p']
    task = data['task']

    # 若 precomputed_pairs 不存在则自动计算（支持从 notebook 直接调用）
    if 'precomputed_pairs' not in data:
        data['precomputed_pairs'] = []
        for j in range(m):
            if task == 'ranking':
                data['precomputed_pairs'].append(
                    ranking_pairs(data['X'][j], data['Y'][j])
                )
            else:
                data['precomputed_pairs'].append(
                    aft_pairs(data['X'][j], data['logTt'][j],
                              data['delta'][j], data['Sigma'])
                )

    theta0_list = []
    for j in range(m):
        if task == 'ranking':
            dX, S = data['precomputed_pairs'][j]
            # SLSQP：精确球面约束求解，替代一阶梯度下降
            th = _slsqp_ranking_init(dX, S, p)
        else:
            dX, dlogTt, r2, r, di, dj, n_val = data['precomputed_pairs'][j]
            gfn = lambda th, dX=dX, dlogTt=dlogTt, r2=r2, r=r, di=di, dj=dj, n=n_val: aft_grad(th, dX, dlogTt, r2, r, di, dj, n)
            lfn = lambda th, dX=dX, dlogTt=dlogTt, r2=r2, r=r, di=di, dj=dj, n=n_val: aft_loss(th, dX, dlogTt, r2, r, di, dj, n)
            init = np.zeros((p, 1))
            th = local_gd(gfn, lfn, init, n_iter=300, lr_init=0.5, project=False)
        theta0_list.append(th)

    theta_naive = np.mean(np.hstack(theta0_list), axis=1, keepdims=True)
    if task == 'ranking':
        theta_naive = _proj_sphere(theta_naive)
    return theta0_list, theta_naive

def compute_agg_grad(j, theta_t_list, data):
    m = data['m']
    W = data['W']
    task = data['task']
    theta_j = theta_t_list[j]

    grad_sum = np.zeros_like(theta_j)
    for l in range(m):
        if W[j, l] > 0:
            if task == 'ranking':
                g = rank_grad(theta_j, *data['precomputed_pairs'][l])
            else:
                g = aft_grad(theta_j, *data['precomputed_pairs'][l])
            grad_sum += W[j, l] * g

    return grad_sum

def inner_admm(theta_t_list, p_t_list, agg_grad_list, H_rho_list, W,
               rho, W_inner, lam_t=0.0, project=False):
    """
    内层广义共识 ADMM
    加入自适应 rho (Residual Balancing) 机制，实现免调参的极速收敛。
    """
    m = W.shape[0]
    p = theta_t_list[0].shape[0]

    nb = [[k for k in range(m) if W[j, k] > 0] for j in range(m)]
    dg = [len(nb[j]) for j in range(m)]

    theta_w = [theta_t_list[j].copy() for j in range(m)]
    p_w = [p_t_list[j].copy() for j in range(m)]

    # 动态计算 omega
    omega = [1.0 / (H_rho_list[j] + 2.0 * rho * dg[j]) for j in range(m)]

    debug_info = {
        'rho_before': rho,
        'omega': omega,
        'inner_theta': [],
        'inner_numerator': [],
        'consensus_gap': [],
        'prim_res': [],  # 记录每步的原始残差
        'dual_res': [],  # 记录每步的对偶残差
        'rho_after': rho
    }

    for _ in range(W_inner):
        theta_prev = [th.copy() for th in theta_w]
        
        # 1. Dual Update (每次内循环先更新对偶变量)
        p_new = []
        consensus_gaps = []
        for j in range(m):
            if nb[j]:
                consensus_gap = sum(theta_w[j] - theta_w[k] for k in nb[j])
            else:
                consensus_gap = np.zeros((p, 1))
            consensus_gaps.append(consensus_gap.copy())
            p_new.append(p_w[j] + rho * consensus_gap)
        p_w = p_new
        debug_info['consensus_gap'] = consensus_gaps

        # 2. Primal Update
        theta_new = []
        numerators = []
        for j in range(m):
            sum_nb = sum(theta_w[k] for k in nb[j]) if nb[j] else np.zeros((p, 1))
            # 重大修改
            # numerator = (
            #         H_rho_list[j] * theta_w[j]  # 使用内层变量作为近端锚点
            #         - agg_grad_list[j]
            #         - p_w[j]
            #         + rho * (dg[j] * theta_w[j] + sum_nb)
            # )
            numerator = (
                    H_rho_list[j] * theta_t_list[j]  # 修复：必须使用外层传入的固定锚点！
                    - agg_grad_list[j]
                    - p_w[j]
                    + rho * (dg[j] * theta_w[j] + sum_nb)
            )
            numerators.append(numerator.copy())
            z_j = omega[j] * numerator

            if lam_t > 0:
                z_j = soft_threshold(z_j, lam_t * omega[j])

            if project: z_j = _proj_sphere(z_j)
            theta_new.append(z_j)

        theta_w = theta_new
        debug_info['inner_theta'].append([th.copy() for th in theta_w])
        debug_info['inner_numerator'].append(numerators)
        
        # --- 3. Residual Balancing (残差平衡机制) ---
        prim_res_sq = 0.0
        dual_res_sq = 0.0
        for j in range(m):
            # 原始残差：当前最新参数的邻居分歧度
            gap_j = sum(theta_w[j] - theta_w[k] for k in nb[j]) if nb[j] else np.zeros((p, 1))
            prim_res_sq += np.sum(gap_j ** 2)
            
            # 对偶残差：本地参数在内循环前后的变化量近似
            dual_res_sq += np.sum((rho * dg[j] * (theta_w[j] - theta_prev[j])) ** 2)
            
        prim_res = np.sqrt(prim_res_sq)
        dual_res = np.sqrt(dual_res_sq)
        
        debug_info['prim_res'].append(prim_res)
        debug_info['dual_res'].append(dual_res)
        
        # 动态调整 rho
        mu = 10.0
        tau = 1.1  # 使用 1.1 平滑调节，防止网络极度稀疏时发生剧烈震荡
        
        rho_changed = False
        if prim_res > mu * dual_res:
            rho = rho * tau
            rho_changed = True
        elif dual_res > mu * prim_res:
            rho = rho / tau
            rho_changed = True
            
        # 惩罚因子一旦改变，必须严格重算依赖于它的 omega 参数
        if rho_changed:
            omega = [1.0 / (H_rho_list[j] + 2.0 * rho * dg[j]) for j in range(m)]
            
    debug_info['rho_after'] = rho

    return theta_w, p_w, rho, debug_info

def compute_ic(theta_list, data, ic_type='bic'):
    m = data['m']
    task = data['task']
    N_total = sum(data['X'][j].shape[0] for j in range(m))
    
    total_loss = 0
    N_pairs_total = 0
    for j in range(m):
        if task == 'ranking':
            dX, S = data['precomputed_pairs'][j]
            total_loss += rank_loss(theta_list[j], dX, S) * len(S)
            N_pairs_total += len(S)
        else:
            dX, dlogTt, r2, r, di, dj, n_val = data['precomputed_pairs'][j]
            pairs_j = n_val * (n_val - 1) / 2
            total_loss += aft_loss(theta_list[j], dX, dlogTt, r2, r, di, dj, n_val) * pairs_j
            N_pairs_total += pairs_j
            
    beta_mat = np.hstack(theta_list)
    beta_avg = np.mean(beta_mat, axis=1)
    df = np.sum(np.abs(beta_avg) > 1e-4) #只有当全局平均后的系数绝对值大于阈值时，才计入自由度 df
    
    avg_loss = total_loss / N_pairs_total if N_pairs_total > 0 else 1e-10
    
    if ic_type.lower() == 'aic':
        ic = np.log(avg_loss) + (2.0 / N_total) * df
    else: # default to bic
        ic = np.log(avg_loss) + (np.log(N_total) / N_total) * df
    return ic

def run_u_admm(data, T=5, W_inner=5, rho=0.1, lam_t=0.0, verbose=False,
               lambda_candidates=None, ic_type='bic', theta0_list=None):
    m, p = data['m'], data['p']
    W_adj = data['W']
    theta_true = data['theta_true']
    task = data['task']

    if 'precomputed_pairs' not in data:
        data['precomputed_pairs'] = []
        for j in range(m):
            if task == 'ranking':
                data['precomputed_pairs'].append(ranking_pairs(data['X'][j], data['Y'][j]))
            else:
                data['precomputed_pairs'].append(aft_pairs(data['X'][j], data['logTt'][j], data['delta'][j], data['Sigma']))

    # 初始化
    if theta0_list is not None:
        init_theta_t = [th.copy() for th in theta0_list]
        theta_naive = np.mean(np.hstack(theta0_list), axis=1, keepdims=True)
        if task == 'ranking':
            theta_naive = _proj_sphere(theta_naive)
    else:
        init_theta_t, theta_naive = init_all_nodes(data)

    # =========================================================
    # 预计算区：标量近似 + 松弛因子极速加速
    # =========================================================
    theoretical_rho_list = []
    H_scale = 25.0  # 核心加速器：将理论最严苛的约束缩小 15 倍

    for j in range(m):
        X_j = data['X'][j]
        cov_j = (X_j.T @ X_j) / X_j.shape[0]
        max_eig = float(np.linalg.eigvalsh(cov_j).max())
        
        # 将上限标量缩小 H_scale 倍，极大提升等效学习率
        rho_j = (max_eig / H_scale) + 1e-3 
        theoretical_rho_list.append(rho_j)

    history = {'rmse': [], 'consensus': [], 'debug': []}

    def _record(th_list):
        rmse = float(np.mean([np.linalg.norm(th_list[j] - theta_true) for j in range(m)]))
        mat = np.hstack(th_list)
        ce = float(np.mean(np.sum((mat - mat.mean(1, keepdims=True))**2, 0)))
        history['rmse'].append(rmse)
        history['consensus'].append(ce)
        return rmse

    if verbose:
        print(f"  [Theory Rho] Mean={np.mean(theoretical_rho_list):.4f}, Max={np.max(theoretical_rho_list):.4f} (After H_scale={H_scale})")

    # =========================================================
    # 阶段一：调参阶段 (绝对外层全局搜索 + 连续热启动)
    # =========================================================
    if lambda_candidates is not None and len(lambda_candidates) > 0:
        best_ic = float('inf')
        best_lam = 0.0
        sorted_lambdas = sorted(lambda_candidates, reverse=True)
        
        # 建立流动的全局热启动起点
        current_theta_start = [th.copy() for th in init_theta_t]
        current_p_start = [np.zeros((p, 1)) for _ in range(m)]
        
        for lam_cand in sorted_lambdas:
            # 每一个 lam 都从上一个 lam 最终收敛的全局状态开始
            th_temp = [th.copy() for th in current_theta_start]
            p_temp = [p.copy() for p in current_p_start]
            rho_temp = rho
            
            # 完整跑完 U-ADMM 的 T 轮代理构建
            for t in range(T):
                th_prev = [th.copy() for th in th_temp]
                
                agg_grad_list = [compute_agg_grad(j, th_temp, data) for j in range(m)]
                th_temp, p_temp, rho_temp, _ = inner_admm(
                    theta_t_list=th_temp, p_t_list=p_temp, agg_grad_list=agg_grad_list,
                    H_rho_list=theoretical_rho_list, W=W_adj, rho=rho_temp,
                    W_inner=W_inner, lam_t=lam_cand, project=(task == 'ranking')
                )
                
                # 提早终止：由于步长松弛，外层通常 2-4 轮即可满足极小差异
                max_diff = max(float(np.linalg.norm(th_temp[j] - th_prev[j])) for j in range(m))
                if max_diff < 1e-4:
                    break
            
            # 使用当前 lam 彻底收敛后的状态计算全局 BIC
            ic_val = compute_ic(th_temp, data, ic_type=ic_type)
            
            if ic_val < best_ic:
                best_ic = ic_val
                best_lam = lam_cand
                
            # 用当前收敛点更新全局热启动池，喂给下一个更小的 lam
            current_theta_start = th_temp
            current_p_start = p_temp
    else:
        best_lam = lam_t

    # =========================================================
    # 阶段二：画图阶段 (使用全局最优 lambda 严格跑满 T 轮，记录完整历史)
    # =========================================================
    theta_t = [th.copy() for th in init_theta_t]
    p_t = [np.zeros((p, 1)) for _ in range(m)]
    current_rho = rho

    r0 = _record(theta_t)
    if verbose:
        print(f'  [t=0 init]  RMSE={r0:.6f}, Selected best_lam={best_lam:.4f}')

    for t in range(T):
        agg_grad_list = [compute_agg_grad(j, theta_t, data) for j in range(m)]
        
        theta_t, p_t, current_rho, debug_info = inner_admm(
            theta_t_list=theta_t, p_t_list=p_t, agg_grad_list=agg_grad_list,
            H_rho_list=theoretical_rho_list, W=W_adj, rho=current_rho,
            W_inner=W_inner, lam_t=best_lam, project=(task == 'ranking')
        )
        
        outer_debug = {
            't': t,
            'theta_t': [th.copy() for th in theta_t],
            'p_t': [p.copy() for p in p_t],
            'lam_t': best_lam,
            'inner_debug': debug_info
        }
        history['debug'].append(outer_debug)

        r = _record(theta_t)
        if verbose:
            print(f'  [t={t+1:2d}]  RMSE={r:.6f}, rho={current_rho:.4f}, lam_t={best_lam:.4f}')

    return theta_t, theta_naive, history