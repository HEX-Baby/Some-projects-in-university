import sympy
import hashlib
import random
from Crypto.Util import number
#生成密钥对
def generate_kay_pair(bit_length=2048):
    #生成p和q，要求模4余3
    while True:
        p=number.getPrime(bit_length//2)
        if p%4==3:
            break
    while True:
        q=number.getPrime(bit_length//2)
        if q%4==3:
            break

    #大整数N
    N=p*q
    #生成密钥
    PK=(p,q)

    return N,PK
#加密
def Encrypt(message,N):
    #message是明文，N和PK就是上面函数定义的
    c=pow(message,2,N)
    return c
#解密
def Decrypt(ciphertext,PK):
    p,q=PK

    #中国剩余定理求解
    #模数列表
    moduli=[p,q]
    #余数列表
    remainders=[[pow(ciphertext,(1+p)//4,p),pow(ciphertext,(1+q)//4,q)],
                [-pow(ciphertext,(1+p)//4,p),pow(ciphertext,(1+q)//4,q)],
                [pow(ciphertext,(1+p)//4,p),-pow(ciphertext,(1+q)//4,q)],
                [-pow(ciphertext,(1+p)//4,p),-pow(ciphertext,(1+q)//4,q)]
                ]
    results=[]
    for i in range(4):
        results.append(sympy.ntheory.modular.crt(moduli,remainders[i])[0])

    return results

if __name__=="__main__":
    λ = int(input("输入安全参数λ="))  # 安全参数λ
    m=int(input('输入明文m='))
    bit_length = 0  # RSA模数
    if λ == 128:
        bit_length = 1024
    elif λ == 192:
        bit_length = 1536
    elif λ == 256:
        bit_length = 2048

    N,PK=generate_kay_pair(bit_length)

    ciphertext=Encrypt(message=m,N=N)

    decipher=Decrypt(ciphertext=ciphertext,PK=PK)

    print('密钥参数：N=%d' % (N))
    print('PK=(p,q)=(%d,%d)' % (PK[0],PK[1]))
    print('明文m=%d' % (m))
    print('密文c=%d' % (ciphertext))
    print('解密后明文列表Decrypt=',decipher)