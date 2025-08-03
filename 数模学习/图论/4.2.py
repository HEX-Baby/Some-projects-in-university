import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 邻接矩阵（节点从 1 号开始编号）
adj_matrix = np.array([
    [0, 1, 4, 0, 0],
    [1, 0, 2, 5, 0],
    [4, 2, 0, 1, 0],
    [0, 5, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

# 创建无向图
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

# 重新编号，使节点从 1 开始（默认是 0 开始）
mapping = {i: i + 1 for i in range(len(adj_matrix))}
G = nx.relabel_nodes(G, mapping)

# 计算从 1 号点到所有点的最短路径
start_node = 1
shortest_paths = nx.single_source_dijkstra_path(G, source=start_node, weight='weight')
shortest_distances = nx.single_source_dijkstra_path_length(G, source=start_node, weight='weight')

# 输出最短路径结果
print("从 1 号点到各点的最短路径：")
for target, path in shortest_paths.items():
    print(f"到 {target} 的路径: {path}，距离: {shortest_distances[target]}")

# 绘制图
pos = nx.spring_layout(G)  # 生成节点布局
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')

# 绘制权重标签
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()