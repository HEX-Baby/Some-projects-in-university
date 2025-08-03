import sympy
import hashlib
import random
from Crypto.Util import number
from sympy.functions.combinatorial.numbers import legendre_symbol
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

    #随机生成t，使得t是p和q的二次非剩余
    while True:
        t=random.randint(1,min(p,q))
        if legendre_symbol(t,p)==-1 and legendre_symbol(t,q)==-1:
            break
    #大整数N
    N=p*q
    #生成公钥
    PK=(N,t)
    #生成私钥
    SK=(p,q)

    return PK,SK

#加密
def Encrypt(message,PK):
    N,t=PK
    #随机选取整数x∈[1,N-1]
    x=random.randint(1,N-1)
    if message==0:
        c=pow(x,2,N)
    else:
        c=pow(x,2,N)*t % N
    #c是密文
    return c

#解密
def Decrypt(c,SK):
    p,q=SK

    x1=legendre_symbol(c,p)
    x2=legendre_symbol(c,q)

    if x1==-1 and x2==-1:
        m=1
    elif x1==1 and x2==1:
        m=0
    return m

if __name__=='__main__':
    λ = int(input("输入安全参数λ="))  # 安全参数λ
    m=int(input('输入明文m{0,1}='))
    bit_length = 0  # RSA模数
    if λ == 128:
        bit_length =1024
    elif λ == 192:
        bit_length = 1536
    elif λ == 256:
        bit_length = 2048

    #生成密钥公钥
    PK,SK=generate_kay_pair(bit_length)

    #加密
    c=Encrypt(message=m,PK=PK)
    #解密
    result=Decrypt(c,SK)

    print('密钥SK=（p,q）=',SK)
    print('公钥PK=（N,t）=',PK)
    print('密文c=',c)
    print('解密后message=',result)