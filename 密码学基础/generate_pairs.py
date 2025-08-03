import random

# 固定密钥字符串转 bit list
key_str = "00111010100101001101011000111111"
key = [int(b) for b in key_str]

# S-box 和 P-box
pi_s = {
    0: 14, 1: 4, 2: 13, 3: 1,
    4: 2, 5: 15, 6: 11, 7: 8,
    8: 3, 9: 10, 10: 6, 11: 12,
    12: 5, 13: 9, 14: 0, 15: 7
}

pi_p = [1, 5, 9, 13, 2, 6, 10, 14,
        3, 7, 11, 15, 4, 8, 12, 16]
pi_p = [i - 1 for i in pi_p]

def s_change(u):
    v = []
    for i in range(4):
        value_before = u[i * 4 + 0] * 8 + u[i * 4 + 1] * 4 + u[i * 4 + 2] * 2 + u[i * 4 + 3]
        value_after = pi_s[value_before]
        v.append(value_after)
    res = []
    for val in v:
        bits = [int(b) for b in bin(val)[2:].zfill(4)]
        res.extend(bits)
    return res

def p_change(v):
    return [v[pi_p[i]] for i in range(16)]

def SPN(x, k):
    w = x
    for r in range(1, 4):
        k_r = k[4 * r - 4 : 4 * r - 4 + 16]
        u = [w[i] ^ k_r[i] for i in range(16)]
        v = s_change(u)
        w = p_change(v)
    u = [w[i] ^ k[12 + i] for i in range(16)]
    v = s_change(u)
    y = [v[i] ^ k[16 + i] for i in range(16)]
    return y

def random_plaintext():
    return [random.randint(0, 1) for _ in range(16)]

# 生成 5000 组 (x1, x2, y1, y2)
results = []
dx = [0,0,0,0,  1,0,1,1,  0,0,0,0,  0,0,0,0]


for _ in range(5000):

    x1 = random_plaintext()
    x2 =[]
    for i in range(len(dx)):
        x2.append(x1[i] ^ dx[i])
    y1 = SPN(x1, key)
    y2 = SPN(x2, key)

    x1_str = ''.join(map(str, x1))
    x2_str = ''.join(map(str, x2))
    y1_str = ''.join(map(str, y1))
    y2_str = ''.join(map(str, y2))

    results.append(f"{x1_str} {x2_str} {y1_str} {y2_str}")

# 写入文件
with open("spn_4tuples.txt", "w") as f:
    f.write('\n'.join(results))

print("已生成 spn_4tuples.txt（5000组16比特明密文四元组）")
