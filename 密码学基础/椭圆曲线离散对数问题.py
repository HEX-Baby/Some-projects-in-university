def ADD(p1, p2, p, a):
    if p1 == (-1, -1):
        return p2
    if p2 == (-1, -1):
        return p1
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and (y1 + y2) % p == 0:
        return (-1, -1)

    if x1 != x2:
        k = (y2 - y1) * pow((x2 - x1), p - 2, p) % p
    else:
        k = (3 * x1 * x1 + a) * pow(2 * y1, p - 2, p) % p

    x3 = (k * k - x1 - x2) % p
    y3 = (k * (x1 - x3) - y1) % p
    return (x3, y3)


if __name__ == '__main__':
    a, b, p = eval(input().replace(' ', ','))
    px, py, qx, qy = eval(input().replace(' ', ','))
    P = (px, py)
    Q = (qx, qy)
    k = 0
    res = (-1, -1)
    while res != Q:
        res = ADD(res, P, p, a)
        k += 1
    print(k)
