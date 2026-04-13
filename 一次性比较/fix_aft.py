# -*- coding: utf-8 -*-
import json
import os

path = r'd:\日常\@课程\科研\高维稀疏\JMLR-2023 U-statistics\algorithms\exp2_one_aft.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'params =' in src or 'params=' in src:
            new_src = '''import sys
import os
sys.path.append(os.path.abspath('.'))
%load_ext autoreload
%autoreload 2
import time
import numpy as np
import matplotlib.pyplot as plt
from models.aft import generate_aft_data
from algorithms.admm import run_u_admm
from utils.excel_utils import append_to_excel
from utils.eval_utils import calculate_metrics, evaluate_correlation

params = {
    'Experiment': 'AFT Survival',
    'm': 10, 
    'n': 200, # ⚠️ 注意：U-统计量是 O(n^2) 复杂度。n=200 的计算量是 n=100 的 4 倍！调参时建议用 100
    'p_prime': 5, 
    'p': 20, 
    'pc': 0.3,
    'T': 40, 
    'W_inner': 5, 
    'rho': 1.3, 
    'ic_type': 'bic', 
    'lambda_candidates': np.logspace(-2.5, -1.5, 10).tolist(),
    'noise_type': 't1',
    'rng_seed': 245,
    'run_baselines': False
}
np.random.seed(params['rng_seed'])

# 2. 生成数据
d_aft = generate_aft_data(
    m=params['m'], n=params['n'], p=params['p'], 
    pc=params['pc'], noise_type=params['noise_type'], rng_seed=params['rng_seed']
)
theta_true = d_aft['theta_true']

# 3. 运行 Proposed (U-ADMM)
t0 = time.time()
theta_u_a, theta_n_a, hist_a = run_u_admm(
    d_aft, T=params['T'], W_inner=params['W_inner'], 
    rho=params['rho'], verbose=True,
    lambda_candidates=params['lambda_candidates'],
    ic_type=params.get('ic_type', 'bic')
)
time_uadmm = time.time() - t0
theta_uadmm = theta_u_a[0]
print(f'Proposed 耗时: {time_uadmm:.1f}s')

# 4. 运行其他基线算法
theta_avg = theta_n_a
rmse_local, rmse_global, rmse_dgd = 0.0, 0.0, 0.0
time_global, time_dgd = 0.0, 0.0

if params['run_baselines']:
    from algorithms.admm import init_all_nodes
    theta0_list, _ = init_all_nodes(d_aft)
    local_rmses = [calculate_metrics(theta_true, th)['RMSE'] for th in theta0_list]
    rmse_local = np.mean(local_rmses)
    
    from algorithms.baselines import run_global_u_erm, run_dgd
    t0 = time.time()
    theta_global = run_global_u_erm(d_aft, lambda_candidates=params['lambda_candidates'], ic_type=params.get('ic_type', 'bic'))
    time_global = time.time() - t0
    rmse_global = calculate_metrics(theta_true, theta_global)['RMSE']
    print(f'Pooled MR 耗时: {time_global:.1f}s')
    
    t0 = time.time()
    theta_dgd = run_dgd(d_aft, T=params['T'] * params['W_inner'], lr=0.1)
    time_dgd = time.time() - t0
    rmse_dgd, _ = calculate_metrics(theta_true, theta_dgd)
    print(f'D-subGD 耗时: {time_dgd:.1f}s')
'''
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Fixed {path}')
