#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;

typedef long long LL;

LL modmul(LL a, LL b, LL p)
{
	a %= p;
	b %= p;
	LL res = 0;
	while (b)
	{
		if (b & 1) res = (res + a) % p;
		b >>= 1;
		a = (2 * a) % p;
	}
	return res;
}

LL qmi(LL a, LL k, LL p)
{
	LL res = 1;
	while (k)
	{
		if (k & 1)res = modmul(res, a, p);
		a = modmul(a, a, p);
		k >>= 1;
	}
	return res;
}

LL exgcd(LL a, LL b, LL& x, LL& y)
{
	if (!b)
	{
		x = 1;
		y = 0;
		return a;
	}
	LL d = exgcd(b, a % b, y, x);
	y -= a / b * x;
	return d;
}

int main()
{
	LL p, q, e, c, d;
	unsigned long long res;
	cin >> p >> q >> e >> c;
	LL y;
	exgcd(e, (p - 1) * (q - 1), d, y);

	d = (d % ((p - 1) * (q - 1)) + (p - 1) * (q - 1)) % ((p - 1) * (q - 1));

	res = qmi(c, d, q * p);

	cout << res;
	return 0;
}
