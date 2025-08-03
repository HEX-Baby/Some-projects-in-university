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