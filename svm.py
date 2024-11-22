import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 设置字体为 SimHei（黑体），确保系统中已安装该字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号


# 读取数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('v1', axis=1)  # 标签列名为'v1'
    y = data['v1']
    return X, y


# 数据预处理
def preprocess_data(X, y):
    # 文本向量化
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X['v2'])  # 使用 'v2' 列进行向量化

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# 训练不同参数的SVM模型
def train_svm_models():
    kernels = ['线性核函数', '高斯核', '多项式核函数']
    C_values = [0.1, 1, 10]
    # 惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
    # C越大，对误分类的惩罚增大，可能导致过拟合；
    # C值小，容错能力增强，泛化能力较强，但也可能欠拟合。
    models = {}

    for kernel in kernels:
        for C in C_values:
            model_name = f'SVM_{kernel}_C={C}'
            models[model_name] = SVC(kernel=kernel, C=C, random_state=42)

    return models


# 评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix


# 可视化混淆矩阵
def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'混淆矩阵 - {model_name}')
    plt.xlabel('预测值')
    plt.ylabel('实际值')
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 加载数据
    file_path = "D:/yikehuj/file_/machine_learn/spam.csv"  #添加数据包路径
    X, y = load_data(file_path)

    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 训练不同模型
    models = train_svm_models()
    results = {}

    # 评估每个模型
    for model_name, model in models.items():
        print(f"\n评估模型: {model_name}")

        # 训练模型
        model.fit(X_train, y_train)

        # 评估模型
        accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
        results[model_name] = accuracy

        print(f"准确率: {accuracy:.4f}")
        print("分类报告:")
        print(report)

        # 绘制混淆矩阵
        plot_confusion_matrix(conf_matrix, model_name)

    # 绘制不同模型的准确率比较
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('不同SVM模型的准确率比较')
    plt.xticks(rotation=45)
    plt.ylabel('准确率')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
