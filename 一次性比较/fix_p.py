import json

path = r'd:\日常\@课程\科研\高维稀疏\JMLR-2023 U-statistics\algorithms\exp2_run_parallel_aft.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if '"p": 20' in src:
            src = src.replace('"p": 20', '"p": 5')
            cell['source'] = [line + '\n' for line in src.split('\n')]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Fixed p in aft')
