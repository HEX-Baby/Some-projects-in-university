import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置 matplotlib 使用的字体为 SimHei（黑体）以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


#===========================
#加载数据，分离特征，划分数据
#===========================
def dataset(file_path):
    df = pd.read_csv(file_path, sep=';')
    x = df.drop(columns=['quality'])
    y = df['quality']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = x.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = x.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    x_preprocessed = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out()# <-- 提取特征名

    x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, y_encoded, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, label_encoder, feature_names

#============================
#定义模型和训练模型
#============================
def train(train_loader):
    x_train, y_train = train_loader
    model = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
    model.fit(x_train, y_train)
    return model

#============================
#模型评估
#============================
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('混淆矩阵热图')
    plt.tight_layout()
    plt.show()

def test(model, test_loader,label_encoder ,feature_names):
    x_test, y_true = test_loader
    y_pred = model.predict(x_test)
    """
    y_pred = model(x_test)
    写法是错的。model 是一个 DecisionTreeClassifier 类的实例，而不是一个函数。你应该使用 .predict() 方法 来进行预测。
    """
    target_names = [str(cls) for cls in label_encoder.classes_]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("✅ 准确率:", acc)
    print("✅ 分类报告:\n", report)
    print("✅ 混淆矩阵:\n", cm)
    plot_confusion_matrix(cm, target_names)
    # 7. 可视化决策树（文本图）
    # ✅ 传入正确的 feature_names（而不是 x_test.columns）
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names,
              class_names=[str(i) for i in np.unique(y_true)],
              filled=True, rounded=True)
    plt.title("决策树结构")
    plt.show()
#=======================================
#主函数
#=======================================
def main():
    file_path = r'E:\python_study\机器学习自学\wine+quality\winequality-red.csv'

    x_train, x_test, y_train, y_test, label_encoder, feature_names = dataset(file_path)

    train_loader = x_train, y_train
    test_loader = x_test, y_test

    model = train(train_loader)
    test(model, test_loader,label_encoder, feature_names)  # <-- 多传一个参数

if __name__ == '__main__':
    main()




