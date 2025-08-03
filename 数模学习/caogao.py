x = input().replace(' ', '')
x = int(x, 2)
x = hex(x)[2:].zfill(8).upper()
print(x)

