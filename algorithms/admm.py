import numpy as np
from utils.math_utils import soft_threshold, _proj_sphere
from models.ranking import rank_grad, rank_loss, rank_hess, ranking_pairs
from models.aft import aft_grad, aft_loss, aft_hess_diag, aft_pairs

def local_gd(grad_fn, loss_fn, init_theta, n_iter=300, lr_init=0.5, project=False):
    """
    带 Armijo 线搜索的梯度下降（无约束，不投影）。
    """
    theta = init_theta.copy()
    for _ in range(n_iter):
        g = grad_fn(theta)
        l0 = loss_fn(theta)
        alpha = lr_init
        for _ in range(25):
            cand = theta - alpha * g
            if project: cand = _proj_sphere(cand)

            step_diff = cand - theta
            if loss_fn(cand) <= l0 + 1e-4 * float(np.sum(g * step_diff)):
                break
            alpha *= 0.5

        theta = theta - alpha * g
        if project: theta = _proj_sphere(theta)
    return theta

def init_all_nodes(data):
    m, p = data['m'], data['p']
    task = data['task']
    theta0_list = []

    for j in range(m):
        if task == 'ranking':
            dX, S = data['precomputed_pairs'][j]
            gfn = lambda th, dX=dX, S=S: rank_grad(th, dX, S)
            lfn = lambda th, dX=dX, S=S: rank_loss(th, dX, S)
            init = np.ones((p, 1)) / np.sqrt(p)
        else:
            dX, dlogTt, r2, r, di, dj, n_val = data['precomputed_pairs'][j]
            gfn = lambda th, dX=dX, dlogTt=dlogTt, r2=r2, r=r, di=di, dj=dj, n=n_val: aft_grad(th, dX, dlogTt, r2, r, di, dj, n)
            lfn = lambda th, dX=dX, dlogTt=dlogTt, r2=r2, r=r, di=di, dj=dj, n=n_val: aft_loss(th, dX, dlogTt, r2, r, di, dj, n)
            init = np.zeros((p, 1))

        th = local_gd(gfn, lfn, init, n_iter=300, lr_init=0.5, project=(task == 'ranking'))
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

    neighbors = [l for l in range(m) if W[j, l] > 0]

    if task == 'ranking':
        grads = [rank_grad(theta_j, *data['precomputed_pairs'][l]) for l in neighbors]
    else:
        grads = [aft_grad(theta_j, *data['precomputed_pairs'][l]) for l in neighbors]

    return np.mean(np.stack(grads, axis=0), axis=0).reshape(-1, 1)

def inner_admm(theta_t_list, agg_grad_list, H_rho_list, W,
               rho, W_inner, lam_t=0.0, project=False):
    """
    内层广义共识 ADMM（无球面投影，对应式 **3）。
    """
    m = W.shape[0]
    p = theta_t_list[0].shape[0]

    nb = [[k for k in range(m) if W[j, k] > 0] for j in range(m)]
    dg = [len(nb[j]) for j in range(m)]

    omega = [1.0 / (H_rho_list[j] + 2.0 * rho * dg[j]) for j in range(m)]

    theta_w = [theta_t_list[j].copy() for j in range(m)]
    p_w = [np.zeros((p, 1)) for _ in range(m)]

    for _ in range(W_inner):
        p_new = []
        for j in range(m):
            if nb[j]:
                consensus_gap = sum(theta_w[j] - theta_w[k] for k in nb[j])
            else:
                consensus_gap = np.zeros((p, 1))
            p_new.append(p_w[j] + rho * consensus_gap)
        p_w = p_new

        theta_new = []
        for j in range(m):
            sum_nb = sum(theta_w[k] for k in nb[j]) if nb[j] else np.zeros((p, 1))

            numerator = (
                (H_rho_list[j] + rho * dg[j]) * theta_w[j]
                - agg_grad_list[j]
                - p_w[j]
                + rho * sum_nb
            )
            z_j = omega[j] * numerator

            if lam_t > 0:
                z_j = soft_threshold(z_j, lam_t * omega[j])

            if project: z_j = _proj_sphere(z_j)
            theta_new.append(z_j)

        theta_w = theta_new

    return theta_w

def run_u_admm(data, T=5, W_inner=5, rho=0.1, lam_t=0.0, verbose=False):
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

    theta_t_local, theta_naive = init_all_nodes(data)
    # 按照论文要求，将初始值设为 naive estimator (所有节点本地估计的均值)
    theta_t = [theta_naive.copy() for _ in range(m)]

    history = {'rmse': [], 'consensus': []}

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

    for t in range(T):
        agg_grad_list = [
            compute_agg_grad(j, theta_t, data)
            for j in range(m)
        ]

        if task == 'ranking':
            H_rho_list = []
            for j in range(m):
                dX, S = data['precomputed_pairs'][j]
                H_j = rank_hess(theta_t[j], dX, S)
                rho_j = float(np.linalg.eigvalsh(H_j).max()) + 1e-3
                H_rho_list.append(rho_j)
        else:
            H_rho_list = []
            for j in range(m):
                dX, dlogTt, r2, r, di, dj, n_val = data['precomputed_pairs'][j]
                rho_j = max(aft_hess_diag(theta_t[j], dX, dlogTt, r2, r, di, dj, n_val), 0.1)
                H_rho_list.append(rho_j)

        theta_t = inner_admm(
            theta_t_list = theta_t,
            agg_grad_list = agg_grad_list,
            H_rho_list = H_rho_list,
            W = W_adj,
            rho = rho,
            W_inner = W_inner,
            lam_t = lam_t,
            project = (task == 'ranking'),
        )

        r = _record(theta_t)
        if verbose:
            print(f'  [t={t+1:2d}]  RMSE={r:.6f}')

    return theta_t, theta_naive, history
