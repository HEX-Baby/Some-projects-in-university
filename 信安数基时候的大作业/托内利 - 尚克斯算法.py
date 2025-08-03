import sympy
import hashlib
import random
from Crypto.Util import number
from sympy.functions.combinatorial.numbers import legendre_symbol

#随机生成一个素数，也可以指定一个
p=number.getPrime(10)

#寻找一个非二次剩余z
z=2
while True:
    if legendre_symbol(z,p)==-1:
        break
    else:
        z+=1

#a为二次剩余
while True:
    a=random.randint(1,p-1)
    if legendre_symbol(a,p)==1:
        break

print('素数p=',p)
print('一个二次剩余a=',a)
print('非二次剩余z=',z)

#分解p-1
temp=p-1
s=0
while temp%2==0:
    s+=1
    temp/=2

q=int(temp)
#初始化变量
M=s                 #当前迭代的阶数
c=pow(z,q,p)        #辅助值
t=pow(a,q,p)        #当前迭代的目标
R=pow(a,(q+1)//2,p) #结果的候选值
print('初始化变量：\nM=(当前迭代的阶数)',M)
print('c = Z^q mod p（辅助值）=',c)
print('t = a^q mod p（当前迭代的目标）=',t)
print('R = a^{(q+1)/2} mod p（结果的候选值）=',R)

#开始迭代
while t%p!=1:
    i=1
    while pow(t,pow(2,i),p)!=1:
        i+=1
    b=pow(c,pow(2,M-i-1),p)
    #更新变量
    R = R*b % p
    t = t*b*b % p
    c=pow(b,2,p)
    M=i
    print('b=c^{2^M-i-1} mod p=',b)
    print('更新后变量：\nM=(当前迭代的阶数)', M)
    print('c = b^2 mod p（辅助值）=', c)
    print('t = t*b^2 mod p（当前迭代的目标）=', t)
    print('R = R*b mod p（结果的候选值）=', R)

#返回结果
print('x^2=a mod p 解为：\n')
print('x=',R,'或x=',p-R)