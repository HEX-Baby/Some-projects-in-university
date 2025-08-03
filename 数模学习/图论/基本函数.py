import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 邻接矩阵，表示图
adj_matrix = np.array([
    [0, 1, 4, 0, 0],
    [1, 0, 2, 5, 0],
    [4, 2, 0, 1, 0],
    [0, 5, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

# 创建图
G = nx.from_numpy_array(adj_matrix)

# 绘制图
pos = nx.spring_layout(G)  # 使用 spring 布局进行节点排布
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')

# 显示图形
plt.show()
'''
无向图：使用 nx.from_numpy_array(adj_matrix) 创建图。
有向图：使用 nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph) 来指定创建有向图。
arrows=True 参数在绘制有向图时启用箭头，表示边的方向。
'''
'''
无向图 (Undirected Graph)
'''
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 无向图的邻接矩阵
adj_matrix = np.array([
    [0, 1, 4, 0, 0],
    [1, 0, 2, 5, 0],
    [4, 2, 0, 1, 0],
    [0, 5, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

# 创建无向图
G = nx.from_numpy_array(adj_matrix)

# 绘制无向图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')

# 获取边的权重
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()

'''
有向图 (Directed Graph)
'''
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
################################################################
# 重新编号，使节点编号从 1 开始
mapping = {i: i + 1 for i in range(len(adj_matrix))}
G = nx.relabel_nodes(G, mapping)
###############################################################

# 绘制有向图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray', arrows=True)

# 获取边的权重
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
#################################################################################################################################
'''
最短路dijkstra
'''
# 重新编号，使节点编号从 1 开始
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
G = nx.from_numpy_array(adj_matrix)

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

'''
如果你想计算从 1 号点到 5 号点的最短路径：
'''
path_1_to_5 = nx.shortest_path(G, source=1, target=5, weight='weight')
distance_1_to_5 = nx.shortest_path_length(G, source=1, target=5, weight='weight')
print(f"从 1 到 5 的最短路径: {path_1_to_5}, 最短距离: {distance_1_to_5}")
'''
如果你想计算有向图的最短路径，只需要改成：
'''
G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

'''
floyd
'''
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
    x[i][n] = sum(x[i])

# 输出更新后的邻接矩阵
print("更新后的最短路径邻接矩阵：")
print(x)
'''
计算并绘制最小生成树
'''
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
'''
着色问题
'''

import networkx as nx
import numpy as np
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

'''
最大流
'''

import networkx as nx
import matplotlib.pyplot as plt

# 创建有向图
G = nx.DiGraph()

# 添加带容量的边
edges = [
    ('s', 'A', 4), ('s', 'B', 3), ('s', 'C', 3), ('s', 'D', 2), ('s', 'E', 4),
    ('A', 1, 1), ('B', 1, 1), ('C', 1, 1), ('D', 1, 1), ('E', 1, 1),
    ('A', '2[1]', 1), ('B', '2[1]', 1), ('C', '2[2]', 1), ('D', '2[2]', 1), ('E', '2[1]', 1),
    ('2[1]', 2, 2), ('2[2]', 2, 2),
    ('A', '3[1]', 1), ('B', '3[2]', 1), ('C', '3[2]', 1), ('D', '3[1]', 1), ('E', '3[2]', 1),
    ('3[1]', 3, 1), ('3[2]', 3, 1),
    ('A', '4[1]', 1), ('B', '4[1]', 1), ('C', '4[1]', 1), ('D', '4[1]', 1),
    ('4[1]', 4, 2), ('E', 4, 1),
    (1, 't', 5), (2, 't', 4), (3, 't', 4), (4, 't', 3),
]

# 向图中添加边，并设置容量
G.add_weighted_edges_from(edges, weight="capacity")

# 设定源点 (source) 和 汇点 (sink)
source, sink = 's', 't'

# 计算最大流
flow_value, flow_dict = nx.maximum_flow(G, source, sink)

# 输出结果
print("最大流值:", flow_value)
print("流动情况:")
for u, v_flow in flow_dict.items():
    for v, flow in v_flow.items():
        if flow > 0:
            print(f"{u} -> {v} : {flow}")


# 绘制图形
plt.figure(figsize=(10, 6))

# 使用层次布局（类似流图的风格）
pos = nx.spring_layout(G, seed=42)  # 自动布局

# 画出图的节点和边
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, edge_color="gray", arrows=True)

# 标注流量/容量
edge_labels = {(u, v): f"{flow}/{G[u][v]['capacity']}" for u, v_flow in flow_dict.items() for v, flow in v_flow.items() if flow > 0}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

# 显示最大流值
plt.title(f"最大流值: {flow_value}", fontsize=14)
plt.show()



'''
最大流矩阵表示
'''



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

# **手动创建有向图**
G = nx.DiGraph()

# 遍历矩阵，添加边
for i in range(len(nodes)):  # 遍历起点
    for j in range(len(nodes)):  # 遍历终点
        if capacity_matrix[i][j] > 0:  # 只有容量 > 0 才添加边
            G.add_edge(nodes[i], nodes[j], capacity=capacity_matrix[i][j])

# 设定源点 (source) 和 汇点 (sink)
source, sink = 's', 't'  # 直接使用字符串节点

# 计算最大流
flow_value, flow_dict = nx.maximum_flow(G, source, sink)

# 输出结果
print("最大流值:", flow_value)
print("流动情况:")
for u, v_flow in flow_dict.items():
    for v, flow in v_flow.items():
        if flow > 0:
            print(f"{u} -> {v} : {flow}")






'''
矩阵表示的最小费用最大流问题
'''
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

'''
（AOA 图 & 关键路径）
'''

import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体 (SimHei) 适用于大部分系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建有向图（AOA）
G = nx.DiGraph()

# 任务及其持续时间（边表示任务）
activities = {
    (1, 2): ('A', 6),
    (1, 3): ('B', 5),
    (2, 4): ('C', 3),
    (4, 5): ('D', 2),
    (5, 6): ('E', 3),
    (3, 7): ('F', 2),
    (6, 7): ('G', 4),
    (7, 8): ('H', 2)
      # 虚拟任务（保持正确拓扑结构）
}

# 添加任务（边）
for (start, end), (task, duration) in activities.items():
    G.add_edge(start, end, task=task, weight=duration)

# 计算关键路径
longest_path = nx.dag_longest_path(G, weight="weight")
longest_duration = nx.dag_longest_path_length(G, weight="weight")

print(f"关键路径：{' -> '.join(map(str, longest_path))}")
print(f"项目总工期：{longest_duration} 天")

# 绘制 AOA 计划网络图
plt.figure(figsize=(8, 5))
pos = nx.spring_layout(G)  # 自动布局
edge_labels = {(i, j): f"{G.edges[i, j]['task']}({G.edges[i, j]['weight']})" for i, j in G.edges}
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="gray", font_size=12, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.title("AOA 计划网络图（边代表任务）", fontsize=14)
plt.show()


# 计算最早开始时间（ES）
ES = {node: 0 for node in G.nodes}  # 初始化所有节点的 ES = 0
for node in nx.topological_sort(G):  # 按拓扑排序顺序遍历
    for pred in G.predecessors(node):  # 遍历所有前序任务
        ES[node] = max(ES[node], ES[pred] + G[pred][node]['weight'])

# 计算最晚完成时间（LF）
project_duration = max(ES.values())  # 项目总工期
LF = {node: project_duration for node in G.nodes}  # 初始化 LF = 项目总工期
for node in reversed(list(nx.topological_sort(G))):  # 反向拓扑排序
    for succ in G.successors(node):  # 遍历所有后续任务
        LF[node] = min(LF[node], LF[succ] - G[node][succ]['weight'])

# 计算最晚开始时间（LS）和松弛时间（TF）
LS = {node: LF[node] for node in G.nodes}  # LS = LF - duration
TF = {node: LF[node] - ES[node] for node in G.nodes}  # 总浮动时间

# 以矩阵格式存储结果
schedule_matrix = []
for (start, end), (task, duration) in activities.items():
    schedule_matrix.append([
        task, start, end, duration,
        ES[start], ES[start] + duration,  # ES, EF
        LF[end] - duration, LF[end],  # LS, LF
        (LF[end] - duration) - ES[start]  # Total Float (TF)
    ])

# 打印结果
print("\n任务调度矩阵（ES, EF, LS, LF, Total Float）")
print("任务  起点  终点  持续时间  ES  EF  LS  LF  TF")
for row in schedule_matrix:
    print("{:<4}  {:<3}  {:<3}  {:<5}  {:<3}  {:<3}  {:<3}  {:<3}  {:<3}".format(*row))


'''
缩短时间的最小费用
'''
import networkx as nx
import matplotlib.pyplot as plt

# 设定 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建有向图（AOA）
G = nx.DiGraph()

# 任务及其持续时间和单位缩短成本（边表示任务）
activities = {
    (1, 2): ('A', 6, 300),  # (起点, 终点): (任务, 持续时间, 每周加速费用)
    (1, 3): ('B', 5, 200),
    (2, 4): ('C', 3, 250),
    (4, 5): ('D', 2, 180),
    (5, 6): ('E', 3, 150),
    (3, 7): ('F', 2, 300),
    (6, 7): ('G', 4, 500),
    (7, 8): ('H', 2, 400)
}

# 添加任务（边）
for (start, end), (task, duration, cost) in activities.items():
    G.add_edge(start, end, task=task, weight=duration, crash_cost=cost)


# 计算关键路径
def compute_critical_path(G):
    return nx.dag_longest_path(G, weight="weight"), nx.dag_longest_path_length(G, weight="weight")


critical_path, project_duration = compute_critical_path(G)
print(f"原始关键路径：{' -> '.join(map(str, critical_path))}")
print(f"原始项目总工期：{project_duration} 天")

# 目标：缩短 1 周，找到最小成本方案
weeks_to_reduce = 1
total_cost = 0

for _ in range(weeks_to_reduce):
    # 计算关键路径
    critical_path, project_duration = compute_critical_path(G)

    # 找到关键路径上 **单位缩短成本最小** 的任务
    min_cost_task = None
    min_cost = float('inf')

    for i in range(len(critical_path) - 1):
        u, v = critical_path[i], critical_path[i + 1]
        if G[u][v]['weight'] > 1:  # 确保还能缩短
            unit_cost = G[u][v]['crash_cost']
            if unit_cost < min_cost:
                min_cost = unit_cost
                min_cost_task = (u, v)

    # 如果没有可缩短的任务，退出
    if min_cost_task is None:
        print("无法继续缩短")
        break

    # 执行缩短
    u, v = min_cost_task
    G[u][v]['weight'] -= 1
    total_cost += G[u][v]['crash_cost']
    print(f"缩短任务 {G[u][v]['task']} 1 天，成本增加 {G[u][v]['crash_cost']} 元")

# 计算最终关键路径
critical_path, project_duration = compute_critical_path(G)
print(f"缩短后关键路径：{' -> '.join(map(str, critical_path))}")
print(f"缩短后项目总工期：{project_duration} 天")
print(f"总加速成本：{total_cost} 元")

# 绘制 AOA 计划网络图
plt.figure(figsize=(8, 5))
pos = nx.spring_layout(G)  # 自动布局
edge_labels = {(i, j): f"{G.edges[i, j]['task']}({G.edges[i, j]['weight']})" for i, j in G.edges}
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="gray", font_size=12,
        font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.title("AOA 计划网络图（缩短工期后）", fontsize=14)
plt.show()
