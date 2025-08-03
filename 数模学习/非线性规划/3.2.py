from sympy.ntheory.modular import crt
"""
这道题是中国剩余定理
"""
# 余数
remainders = [1,0,1,4,3,4,1,0]
# 模数
moduli = [2,3,4,5,6,7,8,9]

# 计算结果
result = crt(moduli, remainders)

print(f"解: {result[0]}")  # 最小正整数解
print(f"模数乘积: {result[1]}")  # 模数的最小公倍数