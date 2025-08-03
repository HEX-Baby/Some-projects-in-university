import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 创建无向图

# 初始化邻接矩阵，表示图的权重（0表示没有边，0表示无穷大）
# 下面示例是一个带权有向图（如果没有边，可以用 0 代替）
inf = float('inf')
g = np.array([
    [0, 2, 7, 0, 0, 0],
    [2, 0, 4, 6, 8, 0],
    [7, 4, 0, 1, 3, 0],
    [0, 6, 1, 0, 1, 6],
    [0, 8, 3, 1, 0, 3],
    [0, 0, 0, 6, 3, 0]
])

# 使用 networkx 创建图
G = nx.DiGraph()  # 创建一个有向图

# 将邻接矩阵中的数据转换为图的边
n = len(g)
for i in range(n):
    for j in range(n):
        if g[i][j] != 0 and i != j:
            G.add_edge(i, j, weight=g[i][j])

# 使用 Floyd-Warshall 算法计算最短路径
shortest_paths = nx.floyd_warshall(G)

x = np.zeros(shape=(6,7))

# 将结果存回邻接矩阵 g 中
for i in range(n):
    for j in range(n):
        # 更新 g[i][j] 为最短路径的值
        x[i][j] = shortest_paths[i][j]

for i in range(n):
    x[i][n] = max(x[i])

# 输出更新后的邻接矩阵
print("更新后的最短路径邻接矩阵：")
print(x)

people=[50, 40, 60, 20, 70, 90]

for i in range(n):
    x[i][n] = 0
    for j in range(len(people)):
        x[i][n] += x[i][j] * people[j]

print(x)