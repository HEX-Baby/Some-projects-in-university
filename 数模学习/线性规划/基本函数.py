from scipy.optimize import linprog
'''
scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs', options=None)
c:目标函数系数，形式为一维数组
A_ub:不等式约束矩阵，A*x <= b
b_ub:不等书右侧约束向量
A_eq:等式约束矩阵，表示 A*x = b
b_eq:等式约束右侧系数向量
bounds:每个变量的上下界，默认为（0，None），即非负约束
method:求解方法，默认为‘highs’，有‘interior-point’、‘revised simplex’等
options:字典，用于设置求解器的详细配置（如容差、最大迭代次数等）。
options={'integer': [0, 1, 1]} 表示的是整数约束的掩码，其中：
    [0, 1, 1] 指定了每个变量在整数限制上的约束：
    第一个变量： 0 表示不要求整数（可以是浮点数）。
    第二个变量： 1 表示该变量 必须是整数。
    第三个变量： 1 也是表示该变量 必须是整数
返回值：

x:最优解
fun:目标函数的最优值(最小值)
success：布尔值，指示是否成功找到最优值
message：描述求解器状态的消息。 
'''

'''
from pulp import LpProblem, LpVariable, LpMinimize, lpSum
from pulp import PULP_CBC_CMD
pulp库
线性规划和混合整数线性规划


1.# 创建问题对象
LpProblem(name, sense)
name：问题名称，可以是任意字符串。
sense：优化目标类型：
    LpMinimize：表示最小化问题。
    LpMaximize：表示最大化问题。

用 LpProblem 创建一个优化问题，可以设置目标为 最小化(LpMinimize) 或 最大化(LpMaximize)：
problem = LpProblem("Example_Problem", LpMinimize)  # 或 LpMaximize


2.#定义决策变量
用 LpVariable 定义决策变量：
LpVariable(name, lowBound=None, upBound=None, cat='Continuous', e=None)

x1 = LpVariable("x1", lowBound=0)  # 非负变量
x2 = LpVariable("x2", lowBound=0, upBound=10)  # 有上下界
x3 = LpVariable("x3", lowBound=0, cat="Integer")  # 整数变量

cat 参数支持：
"Continuous"（默认）：连续变量。
"Integer"：整数变量。
"Binary"：0 或 1 的二元变量。


3.#设置目标函数
# 目标函数：最小化 3x1 + 4x2
problem += 3 * x1 + 4 * x2, "Objective Function"
#注意这里是  +=  

4. #添加约束
用 += 添加约束条件：

# 添加约束条件
problem += 2 * x1 + x2 <= 20, "Constraint 1"
problem += 4 * x1 + 3 * x2 >= 10, "Constraint 2"
problem += x1 + 2 * x2 == 15, "Constraint 3"

5.# 求解
调用 solve() 方法求解
# 求解
problem.solve(PULP_CBC_CMD())  # 默认使用内置的 CBC 求解器
或者problem.solve()

GLPK：适用于小到中等规模的问题，开源且性能不错。
CPLEX：功能强大，支持复杂约束，但为商业软件。
Gurobi：快速高效，支持大规模问题，通常用于商业应用。
COIN-OR CBC：强大的二次规划求解器，开源。

problems.solve(solver='GLPK')  # 使用GLPK求解器
# 或者选择其他求解器，例如：
# problems.solve(solver='CPLEX')
# problems.solve(solver='Gurobi')
# problems.solve(solver='COIN_CMD')

6.输出结果
# 检查求解状态
print("Status:", problem.status)

# 获取最优解
print("x1 =", x1.varValue)
print("x2 =", x2.varValue)

# 获取目标函数值
print("Objective Value =", problem.objective.value())
'''


"""
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpInteger

# 假设已有数据
c = [3, 5]  # 目标函数系数
A_ub = [[1, 2], [3, 1]]  # 不等式约束系数矩阵
b_ub = [6, 9]  # 不等式约束右端
A_eq = [[1, 1]]  # 等式约束系数矩阵
b_eq = [4]  # 等式约束右端

# 定义问题（最大化）
problem = LpProblem("Integer_Programming", LpMaximize)

# 定义变量（整数变量）
num_vars = len(c)  # 变量数量
x = [LpVariable(f"x{i+1}", lowBound=0, cat=LpInteger) for i in range(num_vars)]

# 添加目标函数
problem += lpSum(c[i] * x[i] for i in range(num_vars)), "Objective"

# 添加不等式约束
for i in range(len(A_ub)):
    problem += lpSum(A_ub[i][j] * x[j] for j in range(num_vars)) <= b_ub[i], f"Constraint_ub_{i+1}"

# 添加等式约束
for i in range(len(A_eq)):
    problem += lpSum(A_eq[i][j] * x[j] for j in range(num_vars)) == b_eq[i], f"Constraint_eq_{i+1}"

# 求解
problem.solve()

# 输出结果
print("Optimal Solution:")
for v in x:
    print(f"{v.name} = {v.varValue}")
print("Optimal Value:", problem.objective.value())
"""