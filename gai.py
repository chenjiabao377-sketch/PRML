import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib

# ================= 0. 中文显示设置 =================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# ================= 1. 数据生成函数 =================
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    # 拼接两类数据
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    # 添加噪声
    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels

# ================= 2. 生成训练集和测试集 =================
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)  # 训练集 1000 个点
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)   # 测试集 500 个点

# ================= 3. 定义模型 =================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "AdaBoost + DT": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50
    ),
    "SVM (Linear)": SVC(kernel='linear'),
    "SVM (RBF)": SVC(kernel='rbf'),
    "SVM (Poly)": SVC(kernel='poly', degree=3)
}

results = []
print("正在训练并评估模型...")

# ================= 4. 训练、预测、输出结果 =================
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc})

df_results = pd.DataFrame(results)
print("\n--- 分类性能对比 ---")
print(df_results)

# ================= 5. 可视化 3D 分类结果 =================
for name, clf in models.items():
    y_pred = clf.predict(X_test)

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_test[:, 0], X_test[:, 1], X_test[:, 2],
        c=y_pred, cmap='viridis', marker='o', s=40
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'{name} 分类结果 (3D)')
    plt.show()

# ================= 6. 绘制混淆矩阵 =================
for name, clf in models.items():
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{name} 混淆矩阵')
    plt.show()