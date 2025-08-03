import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#===============
#读取数据，分离特征，划分数据
#===============
def dataset(file_path):
    df = pd.read_excel(file_path)
    x = df.drop(columns=['Concrete compressive strength(MPa, megapascals)'])
    y = df['Concrete compressive strength(MPa, megapascals)']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

#===================
#开始训练
#===================
def train(train_loader):
    x_train, y_train = train_loader
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model
#===================
#评估
#===================
def test(model, test_loader):
    x_test, y_test = test_loader
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    r2 = r2_score(y_pred=y_pred, y_true=y_test)
    a = model.coef_
    b = model.intercept_
    print(f'MSE = {mse}, R^2 = {r2}')
    print(f'系数a = {a}，截距b = {b}')
    return mse, r2
#=====================
#主函数
#=====================
def main():
    file_path = r'E:\python_study\机器学习自学\concrete+compressive+strength\Concrete_Data.xls'
    x_train, x_test, y_train, y_test = dataset(file_path)

    train_loader = x_train, y_train
    test_loader = x_test, y_test

    model = train(train_loader)
    mse, r2 = test(model, test_loader)

if __name__ == '__main__':
    main()