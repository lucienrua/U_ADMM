import numpy as np
import random
from itertools import combinations
from scipy.stats import norm as sp_norm
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
    elif noise_type == 'gumbel':
        return np.random.gumbel(0, 1, size)
    else:
        return np.random.normal(0, 1, size)

def generate_aft_data(m, n, p, pc, cens_target=0.25, noise_type='gumbel', rng_seed=None):
    """
    生成论文 Section 7.2.1 的 AFT（加速失效时间）仿真数据。
    数据生成模型：
        log(T_i) = X_i^T theta* + zeta_i
        X_i ~ N(0, Sigma),  Sigma_kl = 0.5^|k-l|
        zeta_i ~ 指定噪声分布
        C_i ~ Uniform(0, tau)，tau 通过二分查找使删失率 ≈ cens_target
        theta* = (1, 1, ..., 1)^T
    观测量：Ttilde_i = min(T_i, C_i),  delta_i = I(T_i <= C_i)
    Returns
    -------
    dict 包含 m, n, p, theta_true, X, logTt, delta, W, G, Sigma, avg_censoring, noise_type, task
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
        random.seed(rng_seed)

    # 真实参数：前 p_prime 个为 1，其余为 0
    theta_true = np.zeros((p, 1))
    theta_true[:p_prime, 0] = 1.0

    # 协方差矩阵：Sigma_kl = 0.5^|k-l|（Toeplitz 结构）
    Sigma = np.array([[0.5**abs(i - j) for j in range(p)] for i in range(p)])

    # 用先验样本确定删失时间上界 tau
    # 固定先验样本，避免二分搜索过程中随机噪声干扰收敛
    Xp = np.random.multivariate_normal(np.zeros(p), Sigma, 5000)
    logT_p = Xp @ theta_true.flatten() + generate_noise(noise_type, 5000)
    
    # 防止 Cauchy 噪声等产生极值导致 exp 溢出
    logT_p = np.clip(logT_p, -100, 100)
    T_pilot = np.exp(logT_p)

    # 二分查找：寻找 tau 使 P(T_i > C_i) ≈ cens_target
    rng_bisect = np.random.RandomState(42)
    # 使用 95 分位数代替 max，避免极端 outlier 导致 high 过大，二分查找 60 次无法收敛
    low, high = 0.0, float(np.percentile(T_pilot, 95) * 10.0 + 1.0)
    tau = high
    for _ in range(60):
        tau = (low + high) / 2.0
        C_pilot = rng_bisect.uniform(0, tau, 5000)
        current_cens = float(np.mean(T_pilot > C_pilot))

        if current_cens > cens_target:
            low = tau
        else:
            high = tau

    # 生成 ER 网络
    G, W = generate_er_network(m, pc)

    # 各节点独立生成数据
    X_list, logTt_list, delta_list = [], [], []
    for _ in range(m):
        Xj = np.random.multivariate_normal(np.zeros(p), Sigma, n)
        logTj = Xj @ theta_true.flatten() + generate_noise(noise_type, n)
        
        # 同样防止溢出
        logTj = np.clip(logTj, -100, 100)
        Tj = np.exp(logTj)
        
        Cj = np.random.uniform(0, tau, n)
        Ttj = np.minimum(Tj, Cj)
        deltaj = (Tj <= Cj).astype(float)  # 1 = 未删失, 0 = 删失

        X_list.append(Xj)
        logTt_list.append(np.log(np.maximum(Ttj, 1e-10)))  # 防止 log(0)
        delta_list.append(deltaj)

    avg_cens = 1.0 - float(np.mean([d.mean() for d in delta_list]))

    return dict(m=m, n=n, p=p,
                theta_true=theta_true,
                X=X_list, logTt=logTt_list, delta=delta_list,
                W=W, G=G, Sigma=Sigma,
                avg_censoring=avg_cens,
                noise_type=noise_type,
                task='aft')

def aft_pairs(X, logTt, delta, Sigma, base_n=None):
    n = X.shape[0]
    if base_n is None:
        base_n = n
    ii, jj = map(np.array, zip(*combinations(range(n), 2)))
    dX = X[ii] - X[jj]
    dlogTt = logTt[jj] - logTt[ii]
    r2 = np.maximum(np.einsum('ij,jk,ik->i', dX, Sigma, dX) / base_n, 1e-8)
    r = np.sqrt(r2)
    di = delta[ii]
    dj = delta[jj]
    return dX, dlogTt, r2, r, di, dj, n

def aft_grad(theta, dX, dlogTt, r2, r, di, dj, n):
    de = dlogTt + (dX @ theta).flatten()
    z = de / r
    Phi = sp_norm.cdf(z)
    g = (di * Phi).reshape(-1, 1) * dX - (dj * (1 - Phi)).reshape(-1, 1) * dX
    return g.sum(axis=0).reshape(-1, 1) * 2.0 / (n * (n - 1))

def aft_hess_diag(theta, dX, dlogTt, r2, r, di, dj, n):
    de = dlogTt + (dX @ theta).flatten()
    z = de / r
    phi = sp_norm.pdf(z)
    w = (di + dj) * phi / r
    H = (dX.T * w) @ dX * 2.0 / (n * (n - 1))
    return float(np.linalg.eigvalsh(H).max()) + 1e-3

def aft_loss(theta, dX, dlogTt, r2, r, di, dj, n):
    de = dlogTt + (dX @ theta).flatten()
    z = de / r
    term_ij = di * (de * sp_norm.cdf(z) + r * sp_norm.pdf(z))
    term_ji = dj * (-de * sp_norm.cdf(-z) + r * sp_norm.pdf(-z))
    return float(np.sum(term_ij + term_ji) * 2.0 / (n * (n - 1)))
