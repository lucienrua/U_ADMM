import numpy as np

def soft_threshold(x, kappa):
    """
    软阈值算子
    """
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

def _proj_sphere(theta):
    """
    将 theta 投影到 L2 单位球面
    """
    nrm = np.linalg.norm(theta)
    if nrm > 1e-12:
        return theta / nrm
    else:
        # 如果退化，按照真实参数的分布回退，保证 ||theta|| = 1
        p = len(theta)
        v = np.arange(1, p + 1, dtype=float).reshape(p, 1)
        return v / np.linalg.norm(v)
