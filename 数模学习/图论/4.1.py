import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 有向图的邻接矩阵
adj_matrix = np.array([
    [0, 1, 4, 0, 0],
    [0, 0, 2, 5, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
])

# 创建有向图
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
# 重新编号，使节点编号从 1 开始
mapping = {i: i + 1 for i in range(len(adj_matrix))}
G = nx.relabel_nodes(G, mapping)
# 绘制有向图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray', arrows=True)

# 获取边的权重
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()