from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
import sympy
import hashlib
import random
from Crypto.Util import number

#生成密钥对（p，q）,要求模4余3
def generate_prime_with_condition(bits=1024):
    while True:
        p=number.getPrime(bits)
        if p%4==3:
            return p

# 双线性映射函数的模拟，实际情况下需要具体的群和映射定义
def bilinear_map(g, h, x,N):
    return pow(g,x,N)*pow(h,x,N)%N

#生成h
def get_group_element_h(G,q,N):
    u=random.randint(0,N-1)#群G中的随机元素
    g=random.randint(0,N-1)#群G中的随机元素
    h=pow(u,q,N)              #h=u^q(mod N).N是G的阶
    return h,u,g

#生成主密钥MSk,公钥参数PK
def generate_rsa_keypair(bit_length=2048):
    p=generate_prime_with_condition(bit_length//2)
    q=generate_prime_with_condition(bit_length//2)
    #大合数N=pq
    N=p*q
    G=set([random.randint(0,N-1) for _ in range(5)])           #群G是0到N的集合
    G_1=set([random.randint(0,N-1) for _ in range(5)])         #群G_1也是0到N的集合

    h,u,g=get_group_element_h(G,q, N)

    #公钥参数PK
    PK=(N,G,G_1,bilinear_map,g,h)
    #系统主密钥
    MSK=(p,q)
    return PK,MSK

#计算雅可比符号
def Jacobi(a,N):
    return sympy.functions.combinatorial.numbers.jacobi_symbol(a,N)

# 计算 r 值，满足 r^2 ≡ a (mod N) 或 r^2 ≡ -a (mod N)
def get_r(a,N,p,q):
    exp=(N+5-p-q)//8
    r=pow(a,exp,N)
    # 确保 r^2 ≡ a (mod N) 或 r^2 ≡ -a (mod N)
    if pow(r,2,N)!=a:
        r=pow(a,-exp,N)# 使用负指数
    return r

#生成用户密钥
def generate_user_private_key(user_id,N,p,q):
    #计算哈希值a
    a=int(hashlib.sha256(user_id.encode()).hexdigest(),16)%N

    while Jacobi(a,N)!=1:
        # 如果 a 不是二次剩余，则重新计算 a
        a=(a+1)%N# 简单地增加 a，直到满足条件

    r=get_r(a, N, p, q)

    SK_id=(user_id,r,p)
    return SK_id,a

#加密
#公钥参数PK=(N,G,G_1,bilinear_map,g,h)
def Encrypt(PK,a,m):
    N, G, G1, e, g, h = PK
    #明文对应到雅可比值
    def v(m):
        return 1 if m==0 else -1
    e=random.randint(0,1)

    if m==0:
        while True:
            b1=random.randint(1,N-1)
            if Jacobi(b1,N)==1:
                break
        b=(b1+a*sympy.mod_inverse(b1,N))%N
        s1=pow(g,b,N)*pow(h,e,N)
        return s1
    elif m==1:
        while True:
            c1=random.randint(1,N-1)
            if Jacobi(c1,N)==-1:
                break
        c=(c1-a*sympy.mod_inverse(c1,N))%N
        s2=pow(g,c,N)*pow(h,e,N)
        return s2

#解密
def Decrypt(PK,SK_id,ciphertext):
    N, G, G1, e, g, h = PK
    user_id, r, p = SK_id
    s=ciphertext
    # 解密过程（这里简化处理）
    b=sympy.ntheory.discrete_log(pow(g,p,N),pow(s,p,N),N)
    jacobi=Jacobi((b+2*r),N)
    return 0 if jacobi==1 else 1



if __name__=='__main__':
    λ = int(input("输入安全参数λ="))  # 安全参数λ
    userid=input('输入用户ID=')
    m=int(input('输入明文m{0,1}='))
    bit_length = 0  # RSA模数
    if λ == 128:
        bit_length = 512
    elif λ == 192:
        bit_length = 1536
    elif λ == 256:
        bit_length = 2024

    # 生成主密钥MSk,公钥参数PK
    #公钥参数PK=(N,G,G_1,bilinear_map,g,h)
    #系统主密钥MSK=(p,q)
    PK,MSK=generate_rsa_keypair(bit_length)

    # 生成用户密钥
    #SK_id=(user_id,r,p),a为用户ID哈希值
    SK_id,a=generate_user_private_key(user_id=userid,N=PK[0],p=MSK[0],q=MSK[1])

    #加密
    ciphertext=(Encrypt(PK=PK,a=a,m=m))

    #解密
    result=Decrypt(PK=PK,SK_id=SK_id,ciphertext=ciphertext)

    print('公钥系数参数PK(N,G,G_1,e,g,h):')
    print('N=%d' % (PK[0]))
    print('G=', PK[1])
    print('G_1=', PK[2])
    print("e: (a^x * b^x) mod N")
    print('g=%d' % (PK[4]))
    print('h=%d' % (PK[5]))

    print('系统主密钥MSK=(p,q)=(%d,%d):' % (MSK[0],MSK[1]))

    print('用户密钥SK_id=(user_id,r,p):')
    print('user_id=',userid)
    print('r=%d' % SK_id[1])
    print('p=%d' % SK_id[2])

    print('明文m=%d' % m)
    print('密文ciphertext=',ciphertext)

    print('解密后 v^{-1}=%d' % result)






