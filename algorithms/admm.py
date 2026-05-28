import numpy as np
from scipy.optimize import minimize as scipy_minimize
from utils.math_utils import soft_threshold, _proj_sphere
from models.ranking import rank_grad, rank_loss, rank_hess, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_hess_diag, aft_pairs

def local_gd(grad_fn, loss_fn, init_theta, n_iter=20, lr_init=1.0, project=False, lam=0.0, theta_true=None, project_end=False, decay_rate=0.98):
    """
    Proximal Gradient Descent with adaptive Armijo line search and early stopping.
    """
    theta = init_theta.copy()
    history = {'rmse': []}
    if theta_true is not None:
        record_th = _proj_sphere(theta) if project_end else theta
        history['rmse'].append(float(np.linalg.norm(record_th - theta_true)))
        
    alpha = lr_init
    for step in range(n_iter):
        g = grad_fn(theta)
        
        cand = theta
        if project:
            # 对于非凸约束(如球面投影)，固定大步长容易在极小值附近震荡无法收敛。
            # 引入步长衰减，保证其最终能够进入极小区域并触发早停。
            current_alpha = lr_init * (decay_rate ** step)
            cand = theta - current_alpha * g
            if lam > 0:
                cand = soft_threshold(cand, current_alpha * lam)
            cand = _proj_sphere(cand)
        else:
            # Armijo 线搜索自适应步长：初始值略大于前一次迭代的步长
            alpha = min(lr_init, alpha * 1.5)
            l0 = loss_fn(theta)  # 仅在需要线搜索时才计算 loss
            for _ in range(25):
                cand = theta - alpha * g
                if lam > 0:
                    cand = soft_threshold(cand, alpha * lam)

                step_diff = cand - theta
                if loss_fn(cand) <= l0 + float(np.sum(g * step_diff)) + (0.5/alpha) * np.sum(step_diff**2):
                    break
                alpha *= 0.5

        # 在更新历史记录前检查是否达到早停条件
        if np.linalg.norm(cand - theta) < 1e-4:
            theta = cand
            if theta_true is not None:
                record_th = _proj_sphere(theta) if project_end else theta
                history['rmse'].append(float(np.linalg.norm(record_th - theta_true)))
            break
            
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
            
            # 添加经验局部惩罚系数以防止高维过拟合
            n = data['X'][j].shape[0]
            local_lam = 0.1 * np.sqrt(np.log(p) / n)
            th = local_gd(gfn, lfn, init, n_iter=500, lr_init=0.5, project=False, lam=local_lam)
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
    内层广义共识 ADMM (IEEE 2022 公式 (16a)+(17))。
    """
    m = W.shape[0]
    p = theta_t_list[0].shape[0]

    nb = [[k for k in range(m) if W[j, k] > 0] for j in range(m)]
    dg = [len(nb[j]) for j in range(m)]

    theta_w = [theta_t_list[j].copy() for j in range(m)]
    p_w = [p_t_list[j].copy() for j in range(m)]

    omega = [1.0 / (H_rho_list[j] + 2.0 * rho * dg[j]) for j in range(m)]

    for _ in range(W_inner):
        # 1. 对偶更新 (公式 16a)
        p_new = []
        for j in range(m):
            consensus_gap = sum(theta_w[j] - theta_w[k] for k in nb[j]) if nb[j] else np.zeros((p, 1))
            p_new.append(p_w[j] + rho * consensus_gap)
        p_w = p_new

        # 2. 原始更新 (公式 17)
        theta_new = []
        for j in range(m):
            sum_nb = sum(theta_w[k] for k in nb[j]) if nb[j] else np.zeros((p, 1))
            numerator = (
                    H_rho_list[j] * theta_w[j]  # 内层当前迭代锚点 (IEEE 2022 公式(17): ρ_j·β_{v,t})
                    - agg_grad_list[j]
                    - p_w[j]
                    + rho * (dg[j] * theta_w[j] + sum_nb)
            )
            z_j = omega[j] * numerator

            if lam_t > 0:
                z_j = soft_threshold(z_j, lam_t * omega[j])
            if project:
                z_j = _proj_sphere(z_j)
            theta_new.append(z_j)

        theta_w = theta_new

    return theta_w, p_w, rho

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
    else: # 默认使用 BIC 准则
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
                data['precomputed_pairs'].append(
                    aft_pairs(data['X'][j], data['logTt'][j],
                              data['delta'][j], data['Sigma'])
                )

    # 初始化
    if theta0_list is not None:
        theta_t = [th.copy() for th in theta0_list]
        theta_naive = np.mean(np.hstack(theta0_list), axis=1, keepdims=True)
        if task == 'ranking':
            theta_naive = _proj_sphere(theta_naive)
    else:
        theta_t_local, theta_naive = init_all_nodes(data)
        theta_t = [th.copy() for th in theta_t_local]

    # 基于 Proposition 1 预计算各节点理论步长（循环外固定）
    # rho_j > lambda_max(n^-1 * X^T * X)
    theoretical_rho_list = []
    H_scale = 25.0
    for j in range(m):
        X_j = data['X'][j]
        n_j = X_j.shape[0]
        cov_j = (X_j.T @ X_j) / n_j
        rho_j = float(np.linalg.eigvalsh(cov_j).max() / H_scale) + 1e-3
        theoretical_rho_list.append(rho_j)

    p_t = [np.zeros((p, 1)) for _ in range(m)]

    history = {'rmse': [], 'consensus': []}

    def _record(th_list):
        rmse = float(np.mean([np.linalg.norm(th_list[j] - theta_true) for j in range(m)]))
        mat = np.hstack(th_list)
        ce = float(np.mean(np.sum((mat - mat.mean(1, keepdims=True))**2, 0)))
        history['rmse'].append(rmse)
        history['consensus'].append(ce)
        return rmse

    r0 = _record(theta_t)
    if verbose:
        print(f'  [t=0 init]  RMSE={r0:.6f}')
        print(f'  [Theory Rho] Mean={np.mean(theoretical_rho_list):.4f}, Max={np.max(theoretical_rho_list):.4f}')

    current_lam = lam_t
    current_rho = rho

    for t in range(T):
        agg_grad_list = [compute_agg_grad(j, theta_t, data) for j in range(m)]
        H_rho_list = theoretical_rho_list

        if lambda_candidates is not None and len(lambda_candidates) > 0:
            # 逐迭代 lambda 调参：每个外层 t 内，尝试所有 lambda，选 IC 最优
            best_ic = float('inf')
            best_theta_t = None
            best_p_t = None
            best_rho = None
            best_lam = None
            
            for lam_cand in lambda_candidates:
                cand_theta_t, cand_p_t, cand_rho = inner_admm(
                    theta_t_list=theta_t, p_t_list=p_t,
                    agg_grad_list=agg_grad_list,
                    H_rho_list=H_rho_list, W=W_adj,
                    rho=current_rho, W_inner=W_inner,
                    lam_t=lam_cand, project=(task == 'ranking')
                )
                
                ic_val = compute_ic(cand_theta_t, data, ic_type=ic_type)
                if ic_val < best_ic:
                    best_ic = ic_val
                    best_theta_t = cand_theta_t
                    best_p_t = cand_p_t
                    best_rho = cand_rho
                    best_lam = lam_cand
            
            theta_t = best_theta_t
            p_t = best_p_t
            current_rho = best_rho
            current_lam = best_lam
        else:
            theta_t, p_t, current_rho = inner_admm(
                theta_t_list=theta_t, p_t_list=p_t,
                agg_grad_list=agg_grad_list,
                H_rho_list=H_rho_list, W=W_adj,
                rho=current_rho, W_inner=W_inner,
                lam_t=current_lam, project=(task == 'ranking')
            )

        r = _record(theta_t)
        if verbose:
            if lambda_candidates is not None and len(lambda_candidates) > 0:
                print(f'  [t={t+1:2d}]  RMSE={r:.6f}, best_lam={current_lam:.4f}, rho={current_rho:.4f}, {ic_type.upper()}={best_ic:.4f}')
            else:
                print(f'  [t={t+1:2d}]  RMSE={r:.6f}, lam_t={current_lam:.4f}, rho={current_rho:.4f}')

    return theta_t, theta_naive, history