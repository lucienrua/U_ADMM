2. `soft_threshold`：对于输入向量 $x$ 和 threshold $\kappa$，实现软阈值算子：$$\displaystyle [S_{\kappa}(\boldsymbol{x})]_i = \operatorname{sgn}(x_i) (\vert x_i \vert - \kappa)_+\tag{32}$$

3. `_proj_sphere`：针对针对问题 **7.1 Pairwise Ranking Problem**。在训练集中， $\displaystyle \boldsymbol{\theta}^*$ 的估计量 $\displaystyle \widehat{\boldsymbol{\theta}}^*$ 是基于以下逻辑损失函数的经验风险最小化问题的解：
   $$\displaystyle \widehat{\boldsymbol{\theta}}^*=\arg \min_{\Vert \boldsymbol{\theta} \Vert_2 = 1}\mathcal{L}_N(\boldsymbol{\theta}) = \arg \min_{\Vert \boldsymbol{\theta} \Vert_2 = 1}\dfrac{1}{\mathcal{C}^2_N} \sum_{i<j} \ln \left\lbrace 1 + \exp \left[ -\operatorname{sign}(Y_i - Y_j) \boldsymbol{\theta}^\top (\boldsymbol{X}_i - \boldsymbol{X}_j) \right] \right\rbrace\tag{37}$$
   因此需要将 $\boldsymbol{\theta}$ 投影到 $\ell 2$ 单位球面（||theta||\_2 = 1）。
