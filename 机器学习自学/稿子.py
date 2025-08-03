import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# 设置 matplotlib 使用的字体为 SimHei（黑体）以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# =======================
# 自动使用 GPU（如果可用）
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 当前设备：", device)

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class FNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(FNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),

            nn.Linear(hidden2, output_size)
        )
    def forward(self, x):
        return self.model(x)

class Early_stop():
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stopping = False
        self.count = 0
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.count = 0
            self.best_loss = val_loss
            self.early_stopping = False
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stopping = True

def train(model, train_loader, val_loader, criterion, optimizer, epochs=90, patience=10, delta=0):
    early_stopper = Early_stop(patience, delta)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"第{epoch + 1}轮, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        early_stopper(val_loss)
        if early_stopper.early_stopping:
            print(f"⛔ 早停于第 {epoch + 1} 轮，验证损失不再下降。")
            break

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('混淆矩阵热图')
    plt.tight_layout()
    plt.show()

def test(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(predictions)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    report = classification_report(y_true=y_true, y_pred=y_pred)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    print("✅ 准确率:", acc)
    print("✅ 分类报告:\n", report)
    print("✅ 混淆矩阵:\n", cm)

    return acc, report, cm

def main():
    file_path = r'E:\python_study\机器学习自学\bank+marketing\bank\bank-full.csv'

    df = pd.read_csv(file_path, sep=';')
    y = df['y']
    x = df.drop(columns=['y'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_col = x.select_dtypes(include=['object']).columns.tolist()
    numerical_col = x.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_col),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_col)
    ])
    x_preprocess = preprocessor.fit_transform(x)
    target_names = label_encoder.classes_

    x_temp, x_test, y_temp, y_test = train_test_split(x_preprocess, y_encoded, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

    train_dataset = dataset(x_train, y_train)
    val_dataset = dataset(x_val, y_val)
    test_dataset = dataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    output_size = len(np.unique(y_encoded))
    model = FNN(input_size=x_preprocess.shape[1], hidden1=256, hidden2=256, output_size=output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, criterion, optimizer)
    acc, report, cm = test(model, test_loader)

    plot_confusion_matrix(cm, target_names)

if __name__ == '__main__':
    main()
