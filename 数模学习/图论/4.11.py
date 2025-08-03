import networkx as nx
import numpy as np

# 定义节点列表
nodes = ['s', 'A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', 't']
node_index = {node: i for i, node in enumerate(nodes)}  # 生成索引映射

# 定义容量矩阵（capacity_matrix）: 行表示起点，列表示终点
capacity_matrix = np.array([
    #   s  A  B  C  D  E  1  2  3  4  t
    [  0, 4, 3, 3, 2, 4, 0, 0, 0, 0, 0],  # s
    [  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # A
    [  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # B
    [  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # C
    [  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # D
    [  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # E
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],  # 1
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 2
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 3
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],  # 4
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # t
])

# 定义费用矩阵（cost_matrix）
cost_matrix = np.array([
    #   s  A  B  C  D  E  1  2  3  4  t
    [  0, 1, 2, 2, 3, 1, 0, 0, 0, 0, 0],  # s
    [  0, 0, 0, 0, 0, 0, 2, 1, 3, 2, 0],  # A
    [  0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0],  # B
    [  0, 0, 0, 0, 0, 0, 3, 1, 2, 2, 0],  # C
    [  0, 0, 0, 0, 0, 0, 2, 1, 2, 3, 0],  # D
    [  0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 0],  # E
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],  # 2
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 3
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],  # 4
    [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # t
])

# 创建有向图
G = nx.DiGraph()

# 添加边（从矩阵读取）
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if capacity_matrix[i, j] > 0:  # 只有容量大于 0 才添加
            G.add_edge(nodes[i], nodes[j], capacity=capacity_matrix[i, j], weight=cost_matrix[i, j])

# 设定源点 (source) 和 汇点 (sink)
source, sink = 's', 't'

# 计算最小费用最大流
flow_dict = nx.max_flow_min_cost(G, source, sink)  # 计算最小费用最大流
min_cost = nx.cost_of_flow(G, flow_dict)  # 计算总费用
max_flow = sum(flow_dict[source][v] for v in G.successors(source))  # 计算最大流量

# 输出结果
print("最小费用最大流值:", max_flow)
print("最小费用:", min_cost)
print("流动情况:")
for u in flow_dict:
    for v in flow_dict[u]:
        if flow_dict[u][v] > 0:
            print(f"{u} -> {v} : {flow_dict[u][v]}")

# 输出流矩阵
print("流矩阵:")
flow_matrix = np.zeros_like(capacity_matrix)
for u in flow_dict:
    for v in flow_dict[u]:
        if flow_dict[u][v] > 0:
            flow_matrix[node_index[u], node_index[v]] = flow_dict[u][v]
print(flow_matrix)