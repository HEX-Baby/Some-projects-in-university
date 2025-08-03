import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
"""
一切还要从深圳杯的第一问说起，人数为标签，进行分类
"""
##1.自定义数据类
# 1. 检查GPU可用性并设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")
class STRDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    """
    dtype=不能省略
    """
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#2.===================模型定义

class FFNN(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32, class_num=4):
        super(FFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),

            #nn.Dropout(0.5)
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),

            nn.Linear(hidden2, class_num)
            # """
            # 输出层：神经元数量等于类别数量
            # """
        )

    def forward(self, x):
        return self.model(x)

#3
#================
#训练函数
#================

def train(model, loader, criterion, optimizer, epochs=60):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            """
            (outputs, y_batch)顺序不能反过来
            PyTorch的nn.CrossEntropyLoss要求第一个参数是模型的输出（logits），第二个参数是目标标签（targets）
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            #PyTorch的nn.CrossEntropyLoss()默认行为是返回批次的平均损失（mean reduction）
            #size(0)和shape[0]等价,但是size是方法，前面有个点
        total_loss /= len(loader.dataset)
        print(f'第{epoch + 1}轮,损失为：{total_loss:.4f}')

"""
可以加上早停机制
"""

#4
#================
#评估函数
#================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total * 100
    print(f'正确个数为{correct}, 总数为{total}')
    print(f"准确率为{acc:.4f}%")
    return acc
#4
#================
#主函数
#================
def main():
    #read data
    df = pd.read_csv(r'E:\python_study\深圳杯\pythonProject\Q1\Q1_model\主峰数量_40%阈值.csv')
    # 分离特征和标签
    x = df.drop(columns=['人数'])
    y = df['人数']


    # encode (2,3,4,5) -> (0,1,2,3)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    #标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    #划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

    #数据加载
    train_dataset = STRDataset(x_train, y_train)
    test_dataset = STRDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    #模型，损失函数，优化器
    model = FFNN(input_size=x.shape[1])
    criterion = nn.CrossEntropyLoss()
#分类 → CrossEntropyLoss (多分类) / BCEWithLogitsLoss (二分类)
#回归 → MSELoss / SmoothL1Loss
#序列 → CTCLoss
#类别不平衡 → 加权 CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #训练与评估
    train(model, train_loader, criterion, optimizer)
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
