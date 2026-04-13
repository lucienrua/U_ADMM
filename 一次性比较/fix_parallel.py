# -*- coding: utf-8 -*-
import json
import os

def fix_file(path, exp_type):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'params =' in src or 'params=' in src:
                new_src = f'''# ==========================================
# 参数设定 (Hyperparameters)
# ==========================================
NUM_RUNS = 200       # 重复实验次数
NUM_WORKERS = 10     # 进程数

import numpy as np

params = {{
    "Experiment": "{exp_type}",
    "m": 10, 
    "n": 200,
    "p_prime": 5, 
    "p": 20, 
    "pc": 0.3,
    "T": 40, 
    "W_inner": 5, 
    "rho": 1.3, 
    "ic_type": "bic", 
    "lambda_candidates": np.logspace(-2.5, -1.5, 10).tolist(),
    "noise_type": "t1",
    "rng_seed": 245,
    "run_baselines": False 
}}

import os
folder = "ranking" if "Ranking" in params["Experiment"] else "aft"
os.makedirs(folder, exist_ok=True)
filename = f"{{folder}}/exp_{{folder}}_{{NUM_RUNS}}_{{params['noise_type']}}_p{{params['p']}}_pc_{{str(params['pc']).replace('.', '')}}_rho_{{str(params['rho']).replace('.', '')}}.json"

print(f"Starting Parallel Experiments: {{NUM_RUNS}} runs...")
results = []
def update_progress(result):
    results.append(result)
    print(f"\\rProgress: {{len(results)}}/{{NUM_RUNS}} runs completed.", end="", flush=True)

def log_error(e):
    print(f"\\nError in worker process: {{e}}")

from multiprocessing import Pool
from utils.runner import run_single_ranking, run_single_aft

runner_fn = run_single_ranking if "Ranking" in params["Experiment"] else run_single_aft

if __name__ == '__main__':
    with Pool(NUM_WORKERS) as pool:
        for i in range(NUM_RUNS):
            pool.apply_async(runner_fn, args=(i, params), callback=update_progress, error_callback=log_error)
        pool.close()
        pool.join()
    print() 

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({{'parameters': params, 'results': results}}, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {{filename}}")
'''
                cell['source'] = [line + '\n' for line in new_src.split('\n')]
                break

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'Fixed {path}')

fix_file(r'd:\日常\@课程\科研\高维稀疏\JMLR-2023 U-statistics\algorithms\exp1_run_parallel_ranking.ipynb', 'Pairwise Ranking')
fix_file(r'd:\日常\@课程\科研\高维稀疏\JMLR-2023 U-statistics\algorithms\exp2_run_parallel_aft.ipynb', 'AFT Survival')
