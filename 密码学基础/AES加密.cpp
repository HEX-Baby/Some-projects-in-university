#include<iostream>
#include<cstring>
#include<string>
#include<vector>
#include<algorithm>
#include <cstdint>
#include <iomanip>
using namespace std;
//s盒
int s[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};
// 轮常量
uint8_t Rcon[10][4] = {
    {0x01, 0x00, 0x00, 0x00},
    {0x02, 0x00, 0x00, 0x00},
    {0x04, 0x00, 0x00, 0x00},
    {0x08, 0x00, 0x00, 0x00},
    {0x10, 0x00, 0x00, 0x00},
    {0x20, 0x00, 0x00, 0x00},
    {0x40, 0x00, 0x00, 0x00},
    {0x80, 0x00, 0x00, 0x00},
    {0x1b, 0x00, 0x00, 0x00},
    {0x36, 0x00, 0x00, 0x00}
};

// GF有限域下的倍增
uint8_t xtime(uint8_t x)
{
    return (x << 1) ^ ((x & 0x80) ? 0x1b : 0x00);
}
// GF有限域下的乘法
uint8_t gf_mul(uint8_t a, uint8_t b)
{
    uint8_t res = 0;
    while (b)
    {
        if (b & 1)res ^= a;
        a = xtime(a);
        b >>= 1;
    }
    return res;
}
// 将字符串转换为字节
vector<int> init_data(const string& str)
{
	vector<int>res;
	for (int i = 0; i < str.size() - 1; i += 2)
	{
		char temp1 = str[i];
        	char temp2 = str[i + 1];
		
		int t = 0;
        	if (temp1 <= '9' && temp1 >= '0')t += 16 * (temp1 - '0');
        	else if (temp1 >= 'a' && temp1 <= 'f')t += 16 * (10 + temp1 - 'a');
       		if (temp2 <= '9' && temp2 >= '0')t += temp2 - '0';
        	else if (temp2 >= 'a' && temp2 <= 'f')t += 10 + temp2 - 'a';
		res.push_back(t);
	}
	return res;
}
//字节代换
vector<int>subbytes(const vector<int>&a)
{
    vector<int>b;
    for (int i = 0; i < a.size(); i++)
    {
        b.push_back(s[a[i]]);
    }
    return b;
}
// 轮密钥加
vector<int>addroundkey(const vector<int>& a, const vector<int>& key)
{
    vector<int>b;
    for (int i = 0; i < a.size(); i++)
    {
        b.push_back(a[i] ^ key[i]);
    }
    return b;
}
// 行移位
vector<int> rowshift(const vector<int>& a)
{
    vector<int>b;
    int temp1[4][4];
    int temp2[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            temp1[j][i] = a[i * 4 + j];
            temp2[j][i] = a[i * 4 + j];
        }
    }
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            temp1[i][j] = temp2[i][(j + i) % 4];
        }
    }
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            b.push_back(temp1[j][i]);
        }
    }
    return b;
}
// 列混淆
vector<int>mixcolumn(const vector<int>& a)
{
    vector<int>b;
    int state[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            state[j][i] = a[i * 4 + j];
        }
    }
    for (int col = 0; col < 4; col++)
    {
        uint8_t t[4];
        uint8_t u[4];
        for (int i = 0; i < 4; i++)
        {
            t[i] = state[i][col];
        }
        u[0] = gf_mul(t[0], 2) ^ gf_mul(3, t[1]) ^ t[2] ^ t[3];
        u[1] = gf_mul(t[1], 2) ^ gf_mul(3, t[2]) ^ t[3] ^ t[0];
        u[2] = gf_mul(t[2], 2) ^ gf_mul(3, t[3]) ^ t[0] ^ t[1];
        u[3] = gf_mul(t[3], 2) ^ gf_mul(3, t[0]) ^ t[1] ^ t[2];
        for (int i = 0; i < 4; i++)
        {
            state[i][col] = u[i];
        }
    }
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            b.push_back(state[j][i]);
        }
    }
    return b;
}
// 密钥扩展
vector<vector<int>> keyexpand(const vector<int>& a)
{
    vector<vector<int>>key;
    uint8_t state[4][44];
    uint8_t temp[4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            state[j][i] = a[i * 4 + j];
        }
    }
    for (int i = 4; i < 44; i++)
    {
        int Round_count = i / 4;
        if (i % 4 != 0)
        {
            for (int row = 0; row < 4; row++)
            {
                state[row][i] = state[row][i - 1] ^ state[row][i - 4];
            }
        }
        else
        {
            for (int row = 0; row < 4; row++)
            {
                temp[row] = state[(row + 1) % 4][i - 1];
                temp[row] = s[temp[row]];
                temp[row] ^= Rcon[Round_count - 1][row];
            }
            for (int row = 0; row < 4; row++)
            {
                state[row][i] = state[row][i - 4] ^ temp[row];
            }
        }
    }
    vector<int>w;
    for (int i = 0; i < 44; i++)
    {
        for (int row = 0; row < 4; row++)
            w.push_back(state[row][i]);
        if (i % 4 == 3) 
        {
            key.push_back(w);
            w.clear();
        }
    }

    return key;
}
// AES加密
vector<int>AES(const vector<int>& plain, const vector<int>& k)
{
    auto key = keyexpand(k);

    auto b = addroundkey(plain, key[0]);

    for (int i = 1; i <= 9; i++)
    {
        b = subbytes(b);
        b = rowshift(b);
        b = mixcolumn(b);
        b = addroundkey(b, key[i]);
    }

    b = subbytes(b);
    b = rowshift(b);
    b = addroundkey(b, key[10]);

    return b;
}

int main()
{
	string PLAIN, KEY;
    	cin >> KEY >> PLAIN;
	
    	vector<int>plain = init_data(PLAIN);
	vector<int>key = init_data(KEY);
    
    vector<int>cipher = AES(plain, key);
    for (int i = 0; i < cipher.size(); i++)
    {
        cout << uppercase << setfill('0') << setw(2) << hex << cipher[i];
    }
    return 0;
}
