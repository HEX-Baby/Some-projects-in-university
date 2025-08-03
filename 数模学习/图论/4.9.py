import networkx as nx
import numpy as np
from igraph import Graph
import matplotlib.pyplot as plt

adj_matrix = np.array([
    [0, 1, 2, 3, 0],
    [1, 0, 4, 5, 6],
    [2, 4, 0, 7, 8],
    [3, 5, 7, 0, 1],
    [0, 6, 8, 1, 0]
])

# 创建一个图
G = nx.from_numpy_array(adj_matrix)

mapping = {i: i + 1 for i in range(len(adj_matrix))}
G = nx.relabel_nodes(G, mapping)

# 使用贪心算法进行图着色
coloring = nx.greedy_color(G, strategy="largest_first")

# 输出着色结果
print("顶点着色结果:", coloring)

# 计算使用的颜色总数
num_colors = max(coloring.values()) + 1
print("最少需要的颜色数:", num_colors)


# 提取颜色信息
colors = list(coloring.values())

# 绘制图形
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G, seed=42)  # 设置节点布局
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=800, cmap=plt.cm.rainbow, edge_color="gray")
plt.title("Graph Coloring Result")
plt.show()