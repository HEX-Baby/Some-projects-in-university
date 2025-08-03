import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# 设置随机种子以确保结果可重复
np.random.seed(42)

# 顶点数量
n = 10

# 初始化邻接矩阵
adj_matrix = np.zeros((n, n), dtype=int)

# 生成邻接矩阵
for i in range(n):
    for j in range(i + 1, n):  # 避免重复和自环
        if np.random.rand() < 0.6:  # 以0.6的概率生成边
            weight = np.random.randint(1, 11)  # 随机生成[1, 10]的整数权重
            adj_matrix[i][j] = weight
            adj_matrix[j][i] = weight  # 无向图，对称

print("生成的邻接矩阵:")
print(adj_matrix)

G = nx.from_numpy_array(adj_matrix)
mapping = {i: i + 1 for i in range(len(adj_matrix))}
G = nx.relabel_nodes(G, mapping)


# 绘制图
pos = nx.spring_layout(G)  # 使用 spring 布局进行节点排布
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')
# 获取边的权重
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()

start_node = 1
shortest_paths = nx.single_source_dijkstra_path(G, source=start_node, weight='weight')
shortest_distances = nx.single_source_dijkstra_path_length(G, source=start_node, weight='weight')

# 输出最短路径结果
print("从 1 号点到各点的最短路径：")
for target, path in shortest_paths.items():
    print(f"到 {target} 的路径: {path}，距离: {shortest_distances[target]}")


floyd_algorithm = nx.floyd_warshall(G)

x = np.zeros(shape=(n, n))

for i in range(n):
    for j in range(n):
        x[i][j] = floyd_algorithm[i + 1][j + 1]

print(x)
