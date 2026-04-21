"""提取 Cell 1 完整输出和 Cell 2 的 RMSE 数据"""
import json

with open('exp1_one_ranking.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1 - 完整输出
cell1 = nb['cells'][1]
outputs = cell1.get('outputs', [])
full_text = ''
for out in outputs:
    if out.get('output_type') == 'stream':
        full_text += ''.join(out.get('text', []))

# 提取最后部分（DPGD 输出和对比表）
safe = full_text.encode('ascii', errors='replace').decode('ascii')
lines = safe.split('\n')
# 找到 DPGD 相关行和对比表
dpgd_lines = [l for l in lines if 'DPGD' in l or 'D-subGD' in l or 'Algorithm' in l or '---' in l.replace(' ','') or 'Proposed' in l or 'Naive' in l or 'Pooled' in l or 'Local' in l]
print("=== DPGD/DGD 相关输出 ===")
for l in dpgd_lines:
    print(l)

# 完整的最后 50 行
print("\n=== 最后 50 行输出 ===")
for l in lines[-50:]:
    print(l)

# Cell 5 错误
print("\n=== Cell 5 错误详情 ===")
cell5 = nb['cells'][5]
for out in cell5.get('outputs', []):
    if out.get('output_type') == 'error':
        for tb_line in out.get('traceback', []):
            # 去掉 ANSI 转义码
            import re
            clean = re.sub(r'\x1b\[[0-9;]*m', '', tb_line)
            print(clean)
