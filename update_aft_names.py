import json
import os

# 1. 更新 aft_run_parallel.ipynb 的文件名生成
with open('aft_run_parallel.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "filename = f\"{folder}/exp_aft_{NUM_RUNS}_" in line:
                source[i] = 'filename = f"{folder}/{params[\'noise_type\']}_m{params[\'m\']}_n{params[\'n\']}_p{params[\'p\']}_pc{str(params[\'pc\']).replace(\'.\', \'\')}_rho{str(params[\'rho\']).replace(\'.\', \'\')}_cens{str(params[\'cens_target\']).replace(\'.\', \'\')}.json"\\n'

with open('aft_run_parallel.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# 2. 更新 aft_plot.ipynb 的表头
with open('aft_plot.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "| Method | RMSE | MAE | F1 | Prec | Rec | Pairwise_Acc | Time (s) |\\n" in line:
                source[i] = line.replace("Pairwise_Acc", "C-index")

with open('aft_plot.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("AFT filename and table header updated.")
