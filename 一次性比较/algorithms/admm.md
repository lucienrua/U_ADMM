11. `local_gd` 与 `init_all_nodes`：算法的第一步，执行节点本地初始化。各节点 $\displaystyle j$ 独立运行带 Armijo 线搜索的梯度下降，求解本地经验风险 $\displaystyle \mathcal{L}_{j,n}(\boldsymbol{\theta})$，获得初始估计量 $\displaystyle \widehat{\boldsymbol{\theta}}_0^{(j)}$。同时计算不进行网络通信的 Naive 基线估计：$$\displaystyle \widehat{\boldsymbol{\theta}}_{\text{naive}} = \dfrac{1}{K} \sum_{j=1}^K \widehat{\boldsymbol{\theta}}_0^{(j)}$$

12. `compute_agg_grad`：计算去中心化代理梯度（通信步）。在第 $\displaystyle t$ 次外层迭代中，节点 $\displaystyle j$ 收集邻居集合 $\displaystyle \mathcal{A}(j)$ 中各节点的数据视角，评估自身当前参数 $\displaystyle \widehat{\boldsymbol{\theta}}_t^{(j)}$ 处的局部梯度并取平均，构造代理风险梯度：$$\displaystyle \nabla \widetilde{\mathcal{L}}_N^{(j)}(\widehat{\boldsymbol{\theta}}_t^{(j)}) = \dfrac{1}{\vert \mathcal{A}(j) \vert} \sum_{l \in \mathcal{A}(j)} \nabla \mathcal{L}_{l,n}(\widehat{\boldsymbol{\theta}}_t^{(j)})\tag{4,9*}$$

13. `inner_admm`：执行内层广义共识 ADMM 算法。为求解一致性约束下的二次代理优化问题，严格遵循先对偶、后原始的更新次序。辅助对偶变量 $\displaystyle \boldsymbol{p}$ 更新：
    $$\displaystyle \boldsymbol{p}_{w+1}^{(j)} = \boldsymbol{p}_w^{(j)} + {\rho} \sum_{k \in \mathcal{A}(j)} \left( \boldsymbol{\theta}_w^{(j)} - \boldsymbol{\theta}_w^{(k)} \right)\tag{25}$$

原始变量 $\displaystyle \boldsymbol{\theta}$ 更新采用闭式解。计算权重系数 $\displaystyle {\omega}_j = \dfrac{1}{{\rho}_j + 2{\rho} \vert \mathcal{A}(j) \vert}$ 构造中间向量 $\displaystyle \boldsymbol{z}_j$：
$$\displaystyle \boldsymbol{z}_j = {\omega}_j \left( {\rho}_j \boldsymbol{\theta}_w^{(j)} - \nabla \widetilde{\mathcal{L}}_N^{(j)}(\widehat{\boldsymbol{\theta}}_t^{(j)}) - \boldsymbol{p}_{w+1}^{(j)} + {\rho} \sum_{k \in \mathcal{A}(j)} \boldsymbol{\theta}_w^{(k)} \right)\tag{33,c}$$

结合软阈值算子完成本轮更新：
$$\displaystyle \boldsymbol{\theta}_{w+1}^{(j)} = S_{{\lambda}_t {\omega}_j}(\boldsymbol{z}_j)$$

14. `run_u_admm`：整合外层循环与内层优化的完整去中心化算法框架。外层迭代 $\displaystyle t = 0, \dots, T-1$ 轮中，依次触发节点通信计算代理梯度、构造局部 Hessian 近似以确定近端系数 $\displaystyle {\rho}_j$，并调用内层 ADMM 迭代 $\displaystyle W$ 步获取下一轮参数估计 $\displaystyle \widehat{\boldsymbol{\theta}}_{t+1}^{(j)}$。算法逐轮记录各节点参数相对于真实价值的均方根误差用于收敛性分析：$$\displaystyle \operatorname{RMSE} = \dfrac{1}{K} \sum_{j=1}^K \Vert \widehat{\boldsymbol{\theta}}_t^{(j)} - \boldsymbol{\theta}^* \Vert_2$$
