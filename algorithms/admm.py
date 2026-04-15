import numpy as np
from scipy.optimize import minimize as scipy_minimize
from utils.math_utils import soft_threshold, _proj_sphere
from models.ranking import rank_grad, rank_loss, rank_hess, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_hess_diag, aft_pairs

def local_gd(grad_fn, loss_fn, init_theta, n_iter=300, lr_init=0.5, project=False, lam=0.0):
    """
    Proximal Gradient Descent with Armijo line search.
    """
    theta = init_theta.copy()
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
    内层广义共识 ADMM（无球面投影，对应式 **3）。
    加入自适应 rho (Residual Balancing) 机制。
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
        'prim_res': None,
        'dual_res': None,
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

            # 正确的 Proximal Jacobi 迭代分子：
            # H_rho_list[j] * theta_t_list[j] 是 Proximal 锚点，保证子问题有界且强凸
            # 2.0 * rho * sum_nb 是邻居的拉扯力
            numerator = (
                    H_rho_list[j] * theta_w[j]  # 必须使用内层变量作为近端锚点
                    - agg_grad_list[j]
                    - p_w[j]
                    + rho * (dg[j] * theta_w[j] + sum_nb)  # 对应理论中的 rho * sum(theta_j + theta_k)
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

    # 支持通过 theta0_list 传入外部热启动参数
    if theta0_list is not None:
        theta_t = [th.copy() for th in theta0_list]
        theta_naive = np.mean(np.hstack(theta0_list), axis=1, keepdims=True)
        if task == 'ranking':
            theta_naive = _proj_sphere(theta_naive)
    else:
        theta_t_local, theta_naive = init_all_nodes(data)
        theta_t = [th.copy() for th in theta_t_local]

    # 基于 Proposition 1 预计算各节点理论步长（循环外固定，避免逐迭代重算）
    # rho_j > lambda_max(n^-1 * X^T * X)
    theoretical_rho_list = []
    for j in range(m):
        X_j = data['X'][j]
        n_j = X_j.shape[0]
        # 计算本地样本协方差矩阵 n^-1 * X^T * X
        cov_j = (X_j.T @ X_j) / n_j
        # 理论下界：最大特征值 + 1e-3
        rho_j = float(np.linalg.eigvalsh(cov_j).max()) + 1e-3
        theoretical_rho_list.append(rho_j)

    p_t = [np.zeros((p, 1)) for _ in range(m)]  # 初始化对偶变量


    history = {'rmse': [], 'consensus': [], 'debug': []}

    def _record(th_list):
        rmse = float(np.mean([np.linalg.norm(th_list[j] - theta_true)
                               for j in range(m)]))
        mat = np.hstack(th_list)
        ce = float(np.mean(
            np.sum((mat - mat.mean(1, keepdims=True))**2, 0)
        ))
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
        agg_grad_list = [
            compute_agg_grad(j, theta_t, data)
            for j in range(m)
        ]

        # 使用基于理论下界的固定步长
        H_rho_list = theoretical_rho_list

        if lambda_candidates is not None and len(lambda_candidates) > 0:
            best_ic = float('inf')
            best_theta_t = None
            best_p_t = None
            best_rho = None
            best_debug_info = None
            best_lam = None
            
            for lam_cand in lambda_candidates:
                cand_theta_t, cand_p_t, cand_rho, cand_debug_info = inner_admm(
                    theta_t_list = theta_t,
                    p_t_list = p_t,
                    agg_grad_list = agg_grad_list,
                    H_rho_list = H_rho_list,
                    W = W_adj,
                    rho = current_rho,
                    W_inner = W_inner,
                    lam_t = lam_cand,
                    project = (task == 'ranking')
                )
                
                ic_val = compute_ic(cand_theta_t, data, ic_type=ic_type)
                if ic_val < best_ic:
                    best_ic = ic_val
                    best_theta_t = cand_theta_t
                    best_p_t = cand_p_t
                    best_rho = cand_rho
                    best_debug_info = cand_debug_info
                    best_lam = lam_cand
            
            theta_t = best_theta_t
            p_t = best_p_t
            current_rho = best_rho
            debug_info = best_debug_info
            current_lam = best_lam

        else:
            theta_t, p_t, current_rho, debug_info = inner_admm(
                theta_t_list = theta_t,
                p_t_list = p_t,
                agg_grad_list = agg_grad_list,
                H_rho_list = H_rho_list,
                W = W_adj,
                rho = current_rho,
                W_inner = W_inner,
                lam_t = current_lam,
                project = (task == 'ranking')
            )
        
        # Save outer iteration debug info
        outer_debug = {
            't': t,
            'theta_t': [th.copy() for th in theta_t],
            'p_t': [p.copy() for p in p_t],
            'agg_grad_list': [g.copy() for g in agg_grad_list],
            'H_rho_list': H_rho_list.copy(),
            'lam_t': current_lam,
            'inner_debug': debug_info
        }
        if lambda_candidates is not None and len(lambda_candidates) > 0:
            outer_debug['ic_val'] = best_ic
            
        history['debug'].append(outer_debug)

        r = _record(theta_t)
        if verbose:
            if lambda_candidates is not None and len(lambda_candidates) > 0:
                print(f'  [t={t+1:2d}]  RMSE={r:.6f}, best_lam={current_lam:.4f}, rho={current_rho:.4f}, {ic_type.upper()}={best_ic:.4f}')
            else:
                print(f'  [t={t+1:2d}]  RMSE={r:.6f}, lam_t={current_lam:.4f}, rho={current_rho:.4f}')
            

    return theta_t, theta_naive, history
