from collections import Counter

p = [0.082, 0.015, 0.028, 0.043, 0.127,
     0.022, 0.020, 0.061, 0.070, 0.002,
     0.008, 0.040, 0.024, 0.067, 0.075,
     0.019, 0.001, 0.060, 0.063, 0.091,
     0.028, 0.010, 0.023, 0.001, 0.020,
     0.001]
ciphertext = "BNVSNSIHOCEELSSKKYERIFJKXUMBGYKAMQLJTYAVFBKVTDVBPVVRJYYLAOKYMPQSCGDLFSRLLPROYGESEBUUALRWXMMASAZLGLEDFJBZAVVPXWICGJXASCBYEHOSNMULKCEAHTQOKMFLEBKFXLRRFDTZXCIWBJSICBGAWDVYDHAVFJXZIBKCGJIWEAHTTOEWTUHKROVVRGZBXYIREMMASCSPBNLHJMBLRFFJELHWEYLWISTFVVYFJCMHYUYRUFSFMGESIGRLWALSWMNUHSIMYYITCCOPZSICEHBCCMZFEGVJYOCDEMMPGHVAAUMELCMOEHVLTIPSUYILVGFLMVWDVYDBTHFRAYISYSGKVSUUHYHGGCKTMBLRX"

def index_coincidence(strr):
    N = len(strr)
    if N <= 1:
        return 0
    freq = Counter(strr)
    Ic = sum(f * (f - 1) for f in freq.values()) / (N * (N - 1))
    return Ic

def ceaser_shift(text,shift_num):
    res = ''
    for i in text:
        res += chr(((ord(i) - ord('A') - shift_num) % 26 + ord('A')))
    return res

def split_cipher(strr,m):
    return [strr[i::m] for i in range(m)]

def decript(cipher,key,m):
    res =''
    for i,c in enumerate(cipher):
        j = i % m
        res += chr((ord(c) - ord(key[j])) % 26 + ord('A'))
    return res

best_ic=[0]
best_m=0
for i in range(1,8):
    Y = split_cipher(ciphertext,i)
    ic=[]
    for y in Y:
        ic.append(index_coincidence(y))
    print(f'm={i}:',ic)
    x = sum(ic) / i
    if x > sum(best_ic)/len(best_ic):
        best_ic = ic
        best_m=i

print(f"最好的重合指数是：Ic = {best_ic}, 对应的m = {best_m}")
splitcipher = split_cipher(ciphertext,best_m)

k = []
for i in range(best_m):
    cipher = splitcipher[i]
    n = len(cipher)
    Mg = []

    for j in range(26):
        temp_cipher = ceaser_shift(cipher,j)
        f=[]
        for kk in range(26):
            f.append(temp_cipher.count(chr(kk + ord('A'))))
        Mg.append(sum(p[k] * f[k] for k in range(26)) / n)
    k.append(Mg)
print("各个Mg的值：")
for item in k:
    print(item)

key=[]
for i in range(best_m):
    max_num = max(k[i])
    idx = k[i].index(max_num)
    key.append(chr(idx + ord('A')))

print(f"密钥是{key}")
intkey =[]
for i in key:
    x = ord(i) - ord('A')
    intkey.append(x)
print(intkey)

plaintext = decript(ciphertext,key,best_m)
print(f'明文是{plaintext}')
