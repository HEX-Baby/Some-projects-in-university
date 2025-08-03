import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

"""
这个是建筑物水泥强度的那个样本，对强度进行建模回归

这里相对于分类的模板加入了早停机制，具体步骤为：
（1）新增 EarlyStopping 类。
（2）train 函数加入验证集支持和早停判断。
（3）main() 函数拆分出验证集，用于早停判断。
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
#========================
#定义数据集
#========================
class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy().reshape(-1, 1), dtype=torch.float32)  # ✅ 修复错误

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#========================
#定义模型
#========================
class FNN(nn.Module):
    def __init__(self, input_size, hidden1=256, hidden2=128, out_size=1):
        super(FNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),

            nn.Linear(hidden2, out_size)
        )

    def forward(self, x):
        return self.model(x)

#=============================
#早停机制类
#=============================
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.count = 0

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True

#=============================
#训练函数
#=============================
def train(model, train_loader, val_loader, criterion, optimizer, epochs=90, patience=10, delta=0):
    early_stopper = EarlyStopping(patience, delta)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            optimizer.zero_grad()
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)


        #验证机评估和早停
        val_loss = 0
        model.eval()
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"第{epoch + 1}轮, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        early_stopper(val_loss)

        if early_stopper.early_stop:
            print(f"早停于第 {epoch + 1} 轮，验证损失不再下降。")
            break

#================================
#评估函数
#================================
def evaluate(model, loader):
    model.eval()
    print("开始评估：")
    all_pred = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            all_pred.extend(y_pred.numpy())
            all_labels.extend(y_batch.numpy())
            """
            要把tensor变量转换成numpy数组
            """

    R2 = r2_score(y_true=all_labels, y_pred=all_pred)
    MSE = mean_squared_error(y_true=all_labels, y_pred=all_pred)
    print(f'MSE = {MSE:.4f}, R^2 = {R2:.4f}')


#=================================
#主函数部分
#=================================
def main():
    #读取数据
    file_path = r'E:\python_study\机器学习自学\concrete+compressive+strength\Concrete_Data.xls'
    df = pd.read_excel(file_path)

    #分里特征和标签
    x = df.drop(columns=['Concrete compressive strength(MPa, megapascals)'])
    y = df['Concrete compressive strength(MPa, megapascals)']

    #特征标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    #划分训练集，验证集，测试集
    x_temp, x_test, y_temp, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

    #加载数据
    train_dataset = dataset(x_train, y_train)
    val_dataset = dataset(x_validation, y_validation)
    test_dataset = dataset(x_test, y_test)


    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    """
    训练集和测试集的batch大小不一样！！！！！！！
    """
    #模型，损失函数，优化器
    model = FNN(x.shape[1]).to(device)#=================GPU======================

    #注意x的维度===================================！！注意x的维度
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #训练和测试
    train(model, train_loader, val_loader, criterion, optimizer)
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
