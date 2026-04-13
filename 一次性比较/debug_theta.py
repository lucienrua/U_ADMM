import sys
import numpy as np
sys.path.append(r'd:\日常\@课程\科研\高维稀疏\JMLR-2023 U-statistics\algorithms')
from models.ranking import generate_ranking_data
from algorithms.admm import run_u_admm

np.random.seed(95)
d_rank = generate_ranking_data(m=10, n=100, p_prime=5, p=20, pc=0.3, noise_type='t1')
theta_u_r, _, hist = run_u_admm(d_rank, T=40, W_inner=5, rho=3.3,
    lambda_candidates=[1, 0.33, 0.1, 0.033, 0.01, 0.0033], ic_type='bic')

print('Final theta [node 0]:\n', theta_u_r[0].flatten(), flush=True)
print('RMSE=2.0 ??? why?', np.linalg.norm(theta_u_r[0] - d_rank['theta_true']), flush=True)
