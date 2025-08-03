import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 邻接矩阵，表示无向带权图（0 表示没有边）
adj_matrix = np.array([
    [0, 1, 4, 0, 0],
    [1, 0, 2, 5, 0],
    [4, 2, 0, 1, 0],
    [0, 5, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

# 创建无向图
G = nx.from_numpy_array(adj_matrix)

# 重新编号，使节点编号从 1 开始（默认是 0 开始）
mapping = {i: i + 1 for i in range(len(adj_matrix))}
G = nx.relabel_nodes(G, mapping)

# 计算最小生成树 (MST) —— 使用 Kruskal 或 Prim 算法
mst = nx.minimum_spanning_tree(G, weight='weight')

# 绘制原图（浅灰色）和最小生成树（蓝色）
pos = nx.spring_layout(G)  # 生成布局
nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=2000, font_size=12, font_weight='bold', edge_color='gray', alpha=0.5)
nx.draw(mst, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='blue', width=2)

# 获取边的权重并绘制在图上
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# 显示图形
plt.show()

# 输出最小生成树的边
print("最小生成树的边：")
for edge in mst.edges(data=True):
    print(f"{edge[0]} -- {edge[1]} (权重: {edge[2]['weight']})")

print("最短生成树距离：")
print(sum(edge[2]['weight'] for edge in mst.edges(data=True)))

