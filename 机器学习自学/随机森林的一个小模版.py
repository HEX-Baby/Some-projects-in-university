import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 中文显示设置（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""
| 参数             | 说明                        |
| -------------- | ------------------------- |
| `n_estimators` | 决策树的数量，越多越稳定（如 100 或 200） |
| `max_depth`    | 每棵树的最大深度，用于控制过拟合          |
| `n_jobs=-1`    | 多线程并行训练，`-1` 表示使用所有 CPU 核 |
| `random_state` | 保证每次结果一致                  |

"""
#===========================
#加载数据，分离特征，划分数据
#===========================
def dataset(file_path):
    df = pd.read_csv(file_path, sep=';')
    x = df.drop(columns=['quality'])
    y = df['quality']

    categorical_col = x.select_dtypes(include=['object']).columns.tolist()
    numerical_col = x.select_dtypes(exclude=['object']).columns.tolist()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_col),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_col)
    ])

    x_processed = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out()  # <-- 提取特征名

    x_train, x_test, y_train, y_test = train_test_split(x_processed, y_encoded, test_size=0.2, random_state=42)

    """
    LabelEncoder 是把类别映射为整数（顺序由字典序决定）；

    如果用于树模型（如决策树、随机森林），整数编码是没问题的；

    如果用于线性模型或神经网络，建议使用 OneHotEncoder，防止引入假顺序。
    from sklearn.preprocessing import OneHotEncoder
    # 初始化编码器
    encoder = OneHotEncoder(sparse=False)  # sparse=False 返回的是 numpy array
    # 分离分类列和数值列
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(exclude=['object']).columns
    # 拟合并转换
    encoded = encoder.fit_transform(df[categorical_cols])
    # 把结果转换为 DataFrame，列名自动生成
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    # 拼回原数值列
    final_df = pd.concat([encoded_df, df[numerical_cols].reset_index(drop=True)], axis=1)
    """
    return x_train, x_test, y_train, y_test, label_encoder, feature_names

#============================
#自定义模型和训练数据
#============================
def train(train_loader, n_estimators=100, max_depth=None):
    x_train, y_train = train_loader
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model

#===========================
#评估test
#===========================
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('混淆矩阵热图')
    plt.tight_layout()
    plt.show()

def test(model, test_loader, feature_names, label_encoder):
    x_test, y_true = test_loader

    y_pred = model.predict(x_test)

    target_names = [str(cls) for cls in label_encoder.classes_]

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    print("✅ 准确率:", acc)
    print("✅ 分类报告:\n", report)
    print("✅ 混淆矩阵:\n", cm)
    plot_feature_importance(model, feature_names)

#========================
# 特征重要性可视化
#========================
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(" 特征重要性排名（Random Forest）")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

#========================
# 主函数
#========================
def main():
    file_path = r'E:\python_study\机器学习自学\wine+quality\winequality-red.csv'
    x_train, x_test, y_train, y_test, label_encoder, feature_names = dataset(file_path)
    train_loader = x_train, y_train
    test_loader = x_test, y_test
    model = train(train_loader)
    test(model, test_loader, label_encoder, feature_names)

if __name__ == '__main__':
    main()