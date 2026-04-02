import numpy as np
import random
from itertools import combinations
from scipy.special import expit
from utils.network import generate_er_network

def generate_noise(noise_type, size):
    if noise_type == 'normal':
        return np.random.normal(0, 1, size)
    elif noise_type == 'exp':
        return np.random.exponential(1, size) - 1.0
    elif noise_type == 'cauchy':
        return np.random.standard_cauchy(size)
    elif noise_type == 't1':
        return np.random.standard_t(1, size)
    elif noise_type == 't3':
        return np.random.standard_t(3, size)
    else:
        return np.random.normal(0, 1, size)

def generate_ranking_data(m, n, p_prime, p, pc, noise_type='normal', rng_seed=None):
    """
    生成问题一的数据
    Parameters
    ----------
    m        : 节点数
    n        : 每节点样本量
    p_prime  : 非零特征数量
    p        : 特征维数
    pc       : ER 网络密度
    noise_type: 噪声类型 ('normal', 'exp', 'cauchy', 't1', 't3')
    rng_seed : 可选随机种子

    Returns
    -------
    dict 包含 m, n, p, theta_true, X, Y, W, G, quantiles, task
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
        random.seed(rng_seed)

    # p为总特征维数，p_prime 为前置非零特征的数量
    v = np.zeros((p, 1), dtype=float)
    v[:p_prime, 0] = np.arange(1, p_prime + 1)
    theta_true = v / np.linalg.norm(v)

    # 协方差矩阵：对角元为 1，非对角元为 0.5
    Sigma = 0.5 * np.ones((p, p))
    np.fill_diagonal(Sigma, 1.0)

    # 用 50000 个先验样本估计全局分位点（使五个类别等频）
    Xp = np.random.multivariate_normal(np.zeros(p), Sigma, 50_000)
    tp = Xp @ theta_true.flatten() + generate_noise(noise_type, 50_000)
    quantiles = np.percentile(tp, [20, 40, 60, 80])  # 四个切点 t1,...,t4

    # 生成 ER 网络
    G, W = generate_er_network(m, pc)

    # 各节点独立生成数据
    X_list, Y_list = [], []
    for _ in range(m):
        Xj = np.random.multivariate_normal(np.zeros(p), Sigma, n)  # (n, p)
        tj = Xj @ theta_true.flatten() + generate_noise(noise_type, n) # 连续分数
        # searchsorted 实现分段函数 J：返回 {1,2,3,4,5}
        Yj = np.searchsorted(quantiles, tj, side='right') + 1       # (n,)
        X_list.append(Xj)
        Y_list.append(Yj)

    return dict(m=m, n=n, p=p,
                theta_true=theta_true,   # 真实参数 (p,1)
                X=X_list,       # list of m arrays，m个节点，each (n,p)
                Y=Y_list,       # list of m arrays, each (n,) in [5]
                W=W,            # 邻接矩阵 (m,m)
                G=G,            # NetworkX 图
                quantiles=quantiles,     # 全局分位点 (4,)
                noise_type=noise_type,
                task='ranking')          # 任务标识

def ranking_pairs(X, Y):
    """
    预计算节点本地所有有效样本对的差分向量和符号。排除 Y_i = Y_j 的对，加速计算
    """
    n = X.shape[0]
    # 生成所有 C(n,2) 个无序对的索引
    ii, jj = map(np.array, zip(*combinations(range(n), 2)))
    mask = Y[ii] != Y[jj]           # 过滤掉 Y_i == Y_j 的对
    ii, jj = ii[mask], jj[mask]
    dX = X[ii] - X[jj]                # 差分向量 (M, p)
    S = np.sign(Y[ii] - Y[jj]).astype(float)  # 符号 (M,)
    return dX, S

def rank_grad(theta, dX, S):
    """
    返回L的梯度值
    """
    w = -S * expit(-S * (dX @ theta).flatten())               # -s_ij * sigma(-u_ij),       shape (M,)
    return (dX.T @ w).reshape(-1, 1) / len(S)   # (p,1)

def rank_hess(theta, dX, S):
    """
    逻辑排序损失对 theta 的 Hessian（均值形式）。
    """
    u = S * (dX @ theta).flatten()  # u_ij = s_ij * theta^T d_ij
    sig = expit(u)                     # sigma(u_ij)
    w = sig * (1.0 - sig)            # sigma(u)(1-sigma(u)) ∈ (0, 0.25]
    return (dX.T * w) @ dX / len(S)   # (p, p)

def rank_loss(theta, dX, S):
    """
    逻辑排序损失标量值（用于线搜索 / 验证）。
    L = (1/M) * sum log(1 + exp(-u_ij))
    """
    u = S * (dX @ theta).flatten()
    # clip 防止数值溢出
    return float(np.mean(np.log1p(np.exp(-np.clip(u, -30, 30)))))
