# 分布式近端梯度下降 (DPGD) 算法说明

## 1. 算法背景
本代码库处理的任务（Ranking 的成对逻辑损失与 AFT 的高斯平滑秩损失）其经验损失函数 $\mathcal{L}_j(\boldsymbol{\theta})$ 具有**平滑且连续可导**的优良性质。面对 $\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) + \lambda \|\boldsymbol{\theta}\|_1$ 的"平滑+非平滑"复合优化问题，标准的纯次梯度下降（Subgradient Descent）会产生严重的数值振荡且无法保证稀疏结构的精确恢复。
因此，我们引入**分布式近端梯度下降（DPGD, Decentralized Proximal Gradient Descent）**，利用软阈值算子（Soft-thresholding）实现精确的特征截断。

## 2. 数学迭代公式
在每一轮通信迭代 $t$ 中，每个节点 $j$ 执行以下三步解耦操作：

**Step 1: 网络共识 (Consensus)**
节点融合邻居参数，平滑拓扑差异：
$$ \boldsymbol{v}_j^{(t)} = \sum_{k=1}^m W_{jk} \boldsymbol{\theta}_k^{(t)} $$

**Step 2: 本地梯度下降 (Local Gradient Descent)**
利用平滑且可导的经验损失，沿着负梯度方向更新：
$$ \boldsymbol{u}_j^{(t)} = \boldsymbol{v}_j^{(t)} - \alpha \nabla \mathcal{L}_j(\boldsymbol{v}_j^{(t)}) $$

**Step 3: 近端映射 (Proximal Mapping / Soft-thresholding)**
应用软阈值算子处理 $l_1$ 正则化，实现绝对稀疏：
$$ \boldsymbol{\theta}_j^{(t+1)} = \mathcal{S}_{\alpha \lambda}(\boldsymbol{u}_j^{(t)}) = \operatorname{sign}(\boldsymbol{u}_j^{(t)}) \max(|\boldsymbol{u}_j^{(t)}| - \alpha \lambda, 0) $$

**Step 4: 投影与去噪 (Projection & Denoising - 仅限 Ranking 任务)**
$$ \boldsymbol{\theta}_j^{(t+1)} = \Pi_{\mathbb{S}}(\boldsymbol{\theta}_j^{(t+1)}) $$
*工程学修正*：由于球面投影的浮点除法会放大底噪，强行将绝对值小于 $10^{-5}$ 的元素置为 $0$，以消除网络共识带来的涂抹效应，保证 BIC 信息准则评估的准确性。

## 3. 为什么 DPGD 在本实验中表现优异？
1. **彻底分离可导与不可导项**：避免了将损失函数与惩罚项捆绑求次梯度的理论死局。
2. **恒定步长维持压制力**：取消了经典的 $1/\sqrt{t}$ 衰减，使得软阈值惩罚 $\alpha \lambda$ 在后期依然具有强大的截断力度。
3. **物理截断阻断误差传染**：去噪逻辑完美阻断了网络凸组合对稀疏性的破坏（涂抹效应）。

## 4. 工程实现说明 (`run_dpgd` in `algorithms/baselines.py`)

### 4.1 函数签名
```python
def run_dpgd(data, T=500, lr=0.1, lambda_candidates=None,
             ic_type='bic', theta_init_list=None, return_history=False):
```

### 4.2 关键实现细节

| 设计原则 | 实现方式 |
|---|---|
| **冷启动调参** | 每个 `lambda` 候选独立从 `init_theta.copy()` 出发，绝对禁止热启动 |
| **步长策略** | 前 80% 轮次：恒定 `lr`；后 20% 轮次：线性衰减至 `0.5 * lr` |
| **强制跑满** | 无 `max_diff < tol` 的提前终止，所有 `lambda` 候选均跑满 `T` 轮 |
| **近端映射** | `soft_threshold(u_j, lr_t * lam)`，`lr_t * lam` 随步长调度自动缩放 |
| **去噪** | `th_j[np.abs(th_j) < 1e-5] = 0.0`（ranking 任务，proj_sphere 之后） |

### 4.3 与 D-subGD 的对比

| 特性 | D-subGD (`run_dgd`) | DPGD (`run_dpgd`) |
|---|---|---|
| 损失梯度类型 | 次梯度（含 l1 惩罚） | 平滑梯度（不含惩罚） |
| 稀疏化方式 | 软阈值（捆绑次梯度） | 软阈值（独立近端算子） |
| 步长衰减 | $1/\sqrt{t}$（标准理论） | 前段恒定，后段柔性衰减 |
| BIC 去噪 | 无 | 有（`< 1e-5` 物理截断） |
| λ 选择收敛质量 | 中等（步长后期过小） | **更优**（步长压制力强） |
