9. `generate_aft_data`：生成加速失效时间（AFT）删失回归仿真数据。数据生成遵循线性模型 $\displaystyle \log(T_i) = \boldsymbol{X}_i^\top \boldsymbol{\theta}^* + {\zeta}_i$。其中真实参数为全一向量 $\displaystyle \boldsymbol{\theta}^* = (1, \dots, 1)^\top$。特征 $\displaystyle \boldsymbol{X}_i \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})$，协方差矩阵服从 Toeplitz 结构 $\displaystyle \boldsymbol{\Sigma}_{kl} = 0.5^{\vert k-l \vert}$。噪声项 $\displaystyle {\zeta}_i$ 服从标准极值分布 $\displaystyle \operatorname{Gumbel}(0, 1)$。删失时间 $\displaystyle C_i \sim \operatorname{Uniform}(0, \tau)$，通过先验样本调整上限 $\displaystyle \tau$ 以控制目标删失率。最终的观测变量为实际时间与删失时间的极小值 $\displaystyle \widetilde{T}_i = \min(T_i, C_i)$，删失指示函数为 $\displaystyle {\delta}_i = \operatorname{I}(T_i \le C_i)$。

10. `aft_grad` 与 `aft_hess_diag`：实现 Smooth Gehan 损失函数的经验梯度与对角 Hessian 近似。定义残差 $\displaystyle e_i(\boldsymbol{\theta}) = \log \widetilde{T}_i - \boldsymbol{X}_i^\top \boldsymbol{\theta}$，并引入基于特征距离标准化的平滑变量 $\displaystyle z_{ij} = \dfrac{e_j - e_i}{r_{ij}}$，其中带宽 $\displaystyle r_{ij}^2 = \dfrac{(\boldsymbol{X}_i - \boldsymbol{X}_j)^\top \boldsymbol{\Sigma} (\boldsymbol{X}_i - \boldsymbol{X}_j)}{n}$。利用标准正态累积分布函数 $\displaystyle \Phi$ 替代原始的指示函数，损失函数写为：

$$
\begin{align}
	\mathcal{L}_N(\boldsymbol{\theta}) &= \dfrac{1}{\mathcal{C}^2_N} \sum_{i \ne j} {\delta}_i \left[ (e_j (\boldsymbol{\theta}) - e_i(\boldsymbol{\theta})) {\Phi} \left( \dfrac{e_j (\boldsymbol{\theta}) - e_i(\boldsymbol{\theta})}{r_{ij}} \right) + r_{ij} {\phi} \left( \dfrac{e_j (\boldsymbol{\theta}) - e_i(\boldsymbol{\theta})}{r_{ij}} \right)\right]\\ \\
	&= \dfrac{1}{\mathcal{C}^2_N} \sum_{i \ne j} {\delta}_i \left[ \left(\text{常数}- (\boldsymbol{X}_j - \boldsymbol{X}_i)^\top \boldsymbol{\theta} \right) {\Phi} \left( \dfrac{\text{常数}- (\boldsymbol{X}_j - \boldsymbol{X}_i)^\top \boldsymbol{\theta}}{r_{ij}} \right) + r_{ij} {\phi} \left( \dfrac{\text{常数} - (\boldsymbol{X}_j - \boldsymbol{X}_i)^\top \boldsymbol{\theta}}{r_{ij}} \right)\right]
\end{align}\tag{39,6}
$$

其中残差定义为 $\displaystyle e_i(\boldsymbol{\theta}) = \log \widetilde{T}_i - \boldsymbol{X}_i^\top \boldsymbol{\theta}$，带宽为 $\displaystyle r_{ij}^2 = \dfrac{(\boldsymbol{X}_i - \boldsymbol{X}_j)^\top \boldsymbol{\Sigma} (\boldsymbol{X}_i - \boldsymbol{X}_j)}{n}$。
对应的梯度形式为：
$$\displaystyle \nabla \mathcal{L}_N^{(j)}(\boldsymbol{\theta}) = \dfrac{1}{\mathcal{C}_n^2} \sum_{i<j} \left[ {\delta}_i \Phi(z_{ij}) (\boldsymbol{X}_i - \boldsymbol{X}_j) - {\delta}_j (1 - \Phi(z_{ij})) (\boldsymbol{X}_i - \boldsymbol{X}_j) \right]\tag{40}$$
近似 Hessian 矩阵通过取最大特征值获取二次惩罚系数下界，以满足 $\displaystyle {\rho}_j \boldsymbol{I} \succeq \nabla^2 \mathcal{L}_N^{(j)}(\boldsymbol{\theta})$ 的要求：
$$\displaystyle \nabla^2 \mathcal{L}_N^{(j)}(\boldsymbol{\theta}) \approx \dfrac{1}{\mathcal{C}_n^2} \sum_{i<j} ({\delta}_i + {\delta}_j) \dfrac{\phi(z_{ij})}{r_{ij}^2} (\boldsymbol{X}_i - \boldsymbol{X}_j)(\boldsymbol{X}_i - \boldsymbol{X}_j)^\top\tag{41}$$
