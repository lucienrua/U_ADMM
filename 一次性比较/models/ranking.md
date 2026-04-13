4. `generate_ranking_data`：生成论文 Section 7.1.1 中的成对排序仿真数据。基于生成模型
   $$\displaystyle Y_i = J(\boldsymbol{X}_i^\top \boldsymbol{\theta}^* + {\epsilon}_i)$$
   其中真实参数满足 $\displaystyle \Vert \boldsymbol{\theta}^* \Vert_2 = 1$，特征向量 $\displaystyle \boldsymbol{X}_i \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})$，其协方差矩阵对角元为 $\displaystyle 1$，非对角元为 $\displaystyle 0.5$；噪声 $\displaystyle {\epsilon}_i \sim \mathcal{N}(0, 1)$。映射函数 $\displaystyle J$ 利用大规模先验样本的 $\displaystyle 20\%, 40\%, 60\%, 80\%$ 分位点，将连续隐变量分数切分并映射为 $\displaystyle 5$ 个大致等频的有序离散类别。

5. `ranking_pairs`：预先计算并返回本地所有有效的样本对的——差分特征矩阵 $\displaystyle \boldsymbol{d}_{ij} = \boldsymbol{X}_i - \boldsymbol{X}_j$ 以及目标方向符号向量 $\displaystyle s_{ij} = \operatorname{sign}(Y_i - Y_j)$，排除 $Y_i = Y_j$ 的对，加速计算。

6. `rank_grad`：计算逻辑排序损失的经验梯度 $\displaystyle \nabla \mathcal{L}_N^{(j)}(\boldsymbol{\theta})$。输入：节点 $j$ 局部的经验风险由逻辑损失函数构成，对于给定的本地观测数据矩阵 $\displaystyle \boldsymbol{X}$ 和对应的有序离散标签 $\displaystyle \boldsymbol{Y}$，节点局部的经验风险最小化问题中的目标函数为：

   $$
   \begin{align*} \mathcal{L}_N^{(j)}(\boldsymbol{\theta}) &= \dfrac{1}{M} \sum_{i<j, Y_i \neq Y_j} \ln \left\lbrace 1 + \exp \left[ -\operatorname{sign}(Y_i - Y_j) \boldsymbol{\theta}^\top (\boldsymbol{X}_i - \boldsymbol{X}_j) \right] \right\rbrace\\
   &=\dfrac{1}{M} \sum_{i<j, Y_i \neq Y_j} \ln \left\lbrace 1 + \exp \left(-s_{ij}\boldsymbol{\theta}^\top \boldsymbol{d}_{ij} \right)\right\rbrace\end{align*}\tag{37*}
   $$

   其中 $\displaystyle M$ 为满足 $\displaystyle Y_i \neq Y_j$ 的有效样本对数量。
   求关于参数 $\displaystyle \boldsymbol{\theta}$ 的偏导数：
   $$\begin{aligned} \nabla \mathcal{L}_{ij}(\boldsymbol{\theta}) &= \dfrac{\partial}{\partial \boldsymbol{\theta}} \ln \lbrace 1 + \exp(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) \rbrace \\ &= \dfrac{1}{1 + \exp(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})} \cdot \dfrac{\partial}{\partial \boldsymbol{\theta}} \left( 1 + \exp(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) \right) \\ &= \dfrac{1}{1 + \exp(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})} \cdot \exp(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) \cdot (-s_{ij} \boldsymbol{d}_{ij})\\&= \dfrac{1}{\exp(+s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}+1)} \cdot (-s_{ij} \boldsymbol{d}_{ij}) \\&\xlongequal{\sigma(x) = \dfrac{1}{1+\exp(-x)}}- s_{ij}\sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) \boldsymbol{d}_{ij}\end{aligned}$$
   在代码实现中，定义权重列向量 $\displaystyle w_{ij} = -s_{ij} \sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})$。最终，总的经验梯度通过差分矩阵 $\displaystyle \boldsymbol{D}$ 的转置与权重向量 $\displaystyle \boldsymbol{w}$ 的高效矩阵乘法求和并取均值得到：
   $$\displaystyle \nabla \mathcal{L}_N^{(j)}(\boldsymbol{\theta}) = \dfrac{1}{M} \sum_{i<j, Y_i \neq Y_j}  w_{ij} \boldsymbol{d}_{ij} = \dfrac{1}{M} \boldsymbol{D}^\top \boldsymbol{w}$$
   输出：当前 $\displaystyle \boldsymbol{\theta}$ 处的经验梯度向量（维度 $\displaystyle p \times 1$）。

7. `rank_hess`：计算逻辑排序损失的经验 Hessian 矩阵 $\displaystyle \nabla^2 \mathcal{L}_N^{(j)}(\boldsymbol{\theta})$：

$$\displaystyle \begin{aligned} \nabla^2 \mathcal{L}_{ij}(\boldsymbol{\theta}) &= \dfrac{\partial}{\partial \boldsymbol{\theta}^\top} \left[ -s_{ij} \sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) \boldsymbol{d}_{ij} \right] \\ &= -s_{ij} \boldsymbol{d}_{ij} \left[ \dfrac{\partial}{\partial \boldsymbol{\theta}} \sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) \right]^\top \\ &= -s_{ij} \boldsymbol{d}_{ij} \left[ \sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) (1 - \sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})) \cdot (-s_{ij} \boldsymbol{d}_{ij}) \right]^\top \\ &= s_{ij}^2 \sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}) (1 - \sigma(-s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})) \boldsymbol{d}_{ij} \boldsymbol{d}_{ij}^\top \\&\xlongequal{ s_{ij} \in \lbrace -1, 1 \rbrace} \sigma(s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})(1 - \sigma(s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})) \boldsymbol{d}_{ij} \boldsymbol{d}_{ij}^\top\end{aligned}$$

定义二次导数的标量权重列向量为 $\displaystyle \boldsymbol{w}'$，其元素为 $\displaystyle w'_{ij} = \sigma(s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})(1 - \sigma(s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij}))$。经验 Hessian 矩阵：

$$\displaystyle \nabla^2 \mathcal{L}_N^{(j)}(\boldsymbol{\theta}) = \dfrac{1}{M} \sum_{i<j, Y_i \neq Y_j} \sigma(s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})(1 - \sigma(s_{ij} \boldsymbol{\theta}^\top \boldsymbol{d}_{ij})) \boldsymbol{d}_{ij} \boldsymbol{d}_{ij}^\top = \dfrac{1}{M} \boldsymbol{D}^\top \operatorname{diag}(\boldsymbol{w}') \boldsymbol{D}$$

8. `rank_loss`：计算逻辑排序损失的标量值 $\displaystyle \mathcal{L}_N^{(j)}(\boldsymbol{\theta})$： $$\displaystyle \dfrac{1}{M} \sum \ln \lbrace 1 + \exp(-u_{ij}) \rbrace$$。实现时为了避免极端情况下的数值溢出，将内积项限制在 $\displaystyle [-30, 30]$ 范围内。
