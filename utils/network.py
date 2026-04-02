import networkx as nx
import numpy as np
import random

def generate_er_network(m, p_c):
    """
    生成一个 m 节点，密度 p_c的连通随机网络：
    1. 先生成一个随机生成树 (m-1 条边)，保证连通。
    2. 计算目标总边数
    3. 补齐剩余所需的边。
    """
    G = nx.Graph()
    G.add_nodes_from(range(m))

    # --- 1. 生成随机生成树 (保证连通性, m-1 条边) ---
    shuffled_nodes = np.random.permutation(m)
    tree_edges = []
    for i in range(1, m):
        u = shuffled_nodes[i]
        v = shuffled_nodes[np.random.randint(0, i)]
        tree_edges.append((u, v))
    G.add_edges_from(tree_edges)

    # --- 2. 计算需要补充的边数：为保证连通，至少m-1条边
    n_to_add = max(int(m * (m - 1) // 2 * p_c) - (m - 1), 0)

    # --- 3. 补充额外边 ---
    if n_to_add > 0:
        existing_edges = set(tuple(sorted(e)) for e in G.edges())
        available_edges = []

        # 找出所有尚未连接的边
        for u in range(m):
            for v in range(u + 1, m):
                if (u, v) not in existing_edges:
                    available_edges.append((u, v))
        # 无放回抽样
        new_edges = random.sample(available_edges, n_to_add)
        G.add_edges_from(new_edges)

    W = nx.to_numpy_array(G)
    return G, W
