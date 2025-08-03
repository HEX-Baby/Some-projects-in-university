import random
n = eval(input())

k = 0
m = n - 1
a = random.randint(1, n - 1)
while m % 2 == 0:
    k += 1
    m //= 2

b = pow(a, m, n)

def test(b):
    if b == 1:
        print("Yes")
        return
    else:
        for i in range(k):
            if b == n - 1:
                print("Yes")
                return
            else:
                b = pow(b, 2, n)
        print("No")
        return

test(b)
