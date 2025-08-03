x0 = input().split()
key = input().split()

x0 = ''.join(x0)
x0 = list(map(int, x0))

key = ''.join(key)
key = list(map(int, key))

pi_s = {0: 14, 1: 4, 2: 13, 3: 1, 4: 2,
        5: 15, 6: 11, 7: 8, 8: 3, 9: 10,
        10: 6, 11: 12, 12: 5, 13: 9, 14: 0, 15: 7}

pi_p = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
for i in range(16):
    pi_p[i] = pi_p[i] - 1

def s_change(u):
    v= []
    for i in range(4):
        value_before = u[i * 4 + 0] * 8 + u[i * 4 + 1] * 4 + u[i * 4 + 2] * 2 + u[i * 4 + 3]
        value_after = pi_s[value_before]
        v.append(value_after)
    res =[]
    for i in range(4):
        temp_list = [int(b) for b in bin(v[i])[2:].zfill(4)]
        res = res + temp_list
    return res

def p_change(v):
    w = [0] * 16
    for i in range(16):
        w[i] = v[pi_p[i]]

    return w

def SPN(x,k):
    w = x
    v =[]
    for r in range(1, 4):
        k_r = k[4 * r - 4: 4 * r - 4 + 16]
        u = []
        for i in range(16):
            u.append(w[i] ^ k_r[i])

        v = s_change(u)
        w = p_change(v)

    u = []
    for i in range(16):
        u.append(w[i] ^ k[12: 28][i])

    v = s_change(u)

    y = []
    for i in range(16):
        y.append(v[i] ^ k[16: 32][i])

    return y

y0 = SPN(x0, key)
y0 = list(map(str,y0))
y0 = ''.join(y0)
y = y0[:4] + ' ' + y0[4: 8] + ' ' + y0[8: 12] + ' ' + y0[12: 16]
print(y)
