import networkx as nx
import matplotlib.pyplot as plt

# 设定 Matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建有向图（AOA）
G = nx.DiGraph()

# 任务及其持续时间和单位缩短成本（边表示任务）
activities = {
    (1, 2): ('A', 6, 800),  # (起点, 终点): (任务, 持续时间, 每周加速费用)
    (1, 3): ('B', 5, 600),
    (2, 4): ('C', 3, 300),
    (4, 5): ('D', 2, 600),
    (5, 6): ('E', 3, 400),
    (3, 7): ('F', 2, 300),
    (6, 7): ('G', 4, 200),
    (7, 8): ('H', 2, 200)
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
weeks_to_reduce = 3
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
