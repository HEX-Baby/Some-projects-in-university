import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
# 中文显示设置（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#================
#数据预处理
#================
def dataset(file_path):
    df = pd.read_csv(file_path, sep=';')
    # 特征和标签分离
    y = df['y']
    x = df.drop(columns=['y'])
    # 标签编码 yes/no → 1/0
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 自动识别类别特征和数值特征
    categorical_cols = x.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = x.select_dtypes(exclude=['object']).columns.tolist()
    # 使用 ColumnTransformer 对不同类型特征分别处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    x_processor = preprocessor.fit_transform(x)
    """
    | 工具                  | 作用                   |
    | ------------------- | -------------           |
    | `ColumnTransformer` | 对不同列应用不同预处理方法   |
    | `StandardScaler`    | 数值特征标准化             |
    | `OneHotEncoder`     | 类别特征独热编码           |

    """

    x_train, x_test, y_train, y_test = train_test_split(x_processor, y_encoded, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, label_encoder

#====================
#自定义模型和训练模型
#====================
def train(train_loader, kernel='rbf', C=1.0, gamma='scale'):
    x_train, y_train = train_loader
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(x_train, y_train)
    return model
"""
有一种方法可以自动搜索最佳参数
"""

#=====================
#模型评估
#=====================
def test(model, test_loader, label_encoder):
    x_test, y_true = test_loader
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    # 加上类别名称更清晰
    target_names = label_encoder.classes_
    #target_names = [str(cls) for cls in label_encoder.classes_]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("✅ 准确率:", acc)
    print("✅ 分类报告:\n", report)
    print("✅ 混淆矩阵:\n", cm)
    plot_confusion_matrix(cm, label_encoder.classes_)
# #==============
# +✅ 可选改进（多分类可视化）
# 你可以用热图可视化混淆矩阵：
#=======================
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('混淆矩阵热图')
    plt.tight_layout()
    plt.show()
#====================
# 主函数入口
#====================
def main():
    file_path = r'E:\python_study\机器学习自学\bank+marketing\bank\bank-full.csv'
    x_train, x_test, y_train, y_test, label_encoder = dataset(file_path)
    train_loader = x_train, y_train
    test_loader = x_test, y_test
    model = train(train_loader)
    test(model, test_loader, label_encoder)


if __name__ == '__main__':
    main()