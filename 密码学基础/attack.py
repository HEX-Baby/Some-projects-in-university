from collections import defaultdict

# S盒定义及其逆
pi_s = {
    0: 14, 1: 4, 2: 13, 3: 1,
    4: 2, 5: 15, 6: 11, 7: 8,
    8: 3, 9: 10, 10: 6, 11: 12,
    12: 5, 13: 9, 14: 0, 15: 7
}
inv_s = {v: k for k, v in pi_s.items()}

# 读取数据文件
def load_samples(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        samples = []
        for line in lines:
            x1, x2, y1, y2 = line.strip().split()
            samples.append((int(x1, 2), int(x2, 2), int(y1, 2), int(y2, 2)))
        return samples

def main_attack(samples):
    count = [[0] * 16 for _ in range(16)]
    for x1, x2, y1, y2 in samples:
        y1_str = bin(y1)[2:].zfill(16)
        y2_str = bin(y2)[2:].zfill(16)

        # 判断差分路径是否成立（只对 S2 和 S4 做）
        if y1_str[0:4] != y2_str[0:4] or y1_str[8:12] != y2_str[8:12]:
            continue

        y12 = int(y1_str[4:8], 2)  # S2 输出
        y14 = int(y1_str[12:16], 2)  # S4 输出
        y22 = int(y2_str[4:8], 2)
        y24 = int(y2_str[12:16], 2)

        for L1 in range(16):
            for L2 in range(16):
                v12 = y12 ^ L1
                v14 = y14 ^ L2
                v22 = y22 ^ L1
                v24 = y24 ^ L2

                u12 = inv_s[v12]
                u14 = inv_s[v14]
                u22 = inv_s[v22]
                u24 = inv_s[v24]

                du2 = u12 ^ u22
                du4 = u14 ^ u24

                if du2 == 0b0110 and du4 == 0b0110:
                    count[L1][L2] += 1

    max_val = -1
    maxkey = ()
    for L1 in range(16):
        for L2 in range(16):
            print(f"L1={L1:04b}, L2={L2:04b}, count={count[L1][L2]}")

            if count[L1][L2] > max_val:
                max_val = count[L1][L2]
                maxkey = (format(L1, '04b'), format(L2, '04b'))
    return maxkey



samples = load_samples("spn_4tuples.txt")
maxkey = main_attack(samples)
print(maxkey)

