import json

def update_plot(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'avg_hist = np.mean([r[\'Proposed\'][\'hist_rmse\'] for r in res], axis=0)' in source:
                new_source = []
                for line in cell['source']:
                    if "avg_hist = None" in line:
                        new_source.extend([
                            "    avg_hist = None\n",
                            "    pooled_hist = None\n",
                            "    dgd_hist = None\n"
                        ])
                    elif "if method == 'Proposed' and 'hist_rmse' in res[0]['Proposed']:" in line:
                        new_source.append(line)
                    elif "avg_hist = np.mean([r['Proposed']['hist_rmse'] for r in res], axis=0)" in line:
                        new_source.extend([
                            line,
                            "            if method == 'Pooled' and 'hist_rmse' in res[0]['Pooled']:\n",
                            "                pooled_hist = np.mean([r['Pooled']['hist_rmse'] for r in res], axis=0)\n",
                            "            if method == 'D-subGD' and 'hist_rmse' in res[0]['D-subGD']:\n",
                            "                dgd_hist = np.mean([r['D-subGD']['hist_rmse'] for r in res], axis=0)\n"
                        ])
                    elif "avg_pooled_rmse = mean_rmse" in line:
                        new_source.append(line)
                    elif "plt.hlines(avg_pooled_rmse, xmin=0, xmax=total_steps, color='m', linestyle=':', label=f'Pooled MR (RMSE={avg_pooled_rmse:.4f})')" in line:
                        new_source.extend([
                            "        if pooled_hist is not None:\n",
                            "            x_pooled = np.arange(len(pooled_hist))\n", # Assuming default tick matching
                            "            plt.plot(x_pooled, pooled_hist, color='m', linestyle=':', label=f'Pooled MR')\n",
                            "        elif avg_pooled_rmse is not None:\n",
                            "            plt.hlines(avg_pooled_rmse, xmin=0, xmax=total_steps, color='m', linestyle=':', label=f'Pooled MR (RMSE={avg_pooled_rmse:.4f})')\n"
                        ])
                    elif "plt.hlines(avg_dgd_rmse, xmin=0, xmax=total_steps, color='c', linestyle='-', label=f'D-subGD (RMSE={avg_dgd_rmse:.4f})')" in line:
                        new_source.extend([
                            "        if dgd_hist is not None:\n",
                            "            x_dgd = np.arange(len(dgd_hist))\n",       # Assuming default tick matching
                            "            plt.plot(x_dgd, dgd_hist, color='c', linestyle='-', label=f'D-subGD')\n",
                            "        elif avg_dgd_rmse is not None:\n",
                            "            plt.hlines(avg_dgd_rmse, xmin=0, xmax=total_steps, color='c', linestyle='-', label=f'D-subGD (RMSE={avg_dgd_rmse:.4f})')\n"
                        ])
                    else:
                        new_source.append(line)
                        
                cell['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

update_plot('/U_ADMM-main/exp1_plot_ranking.ipynb')
update_plot('/U_ADMM-main/exp2_plot_aft.ipynb')
