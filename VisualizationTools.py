import numpy as np
import matplotlib.pyplot as plt

from QSVM import QSVM

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_QSVM_decision_boundary(dataset_list, qsvm: QSVM):
    """
    绘制 QSVM 决策边界（仅支持二维特征）。
    
    参数
    ----
    dataset_list : (X_train, y_train, X_test, y_test_or_pred)
        - X_train: 训练集特征，形状 (n_train, 2)
        - y_train: 训练集真实标签，形状 (n_train,)
        - X_test:  测试集特征，形状 (n_test, 2)
        - y_test_or_pred: 测试集真实标签 或 预测标签
          * 如果你传的是 y_test（真实标签），函数会自动在内部调用 qsvm 预测；
          * 如果你传的是 y_pred（预测标签），函数会用它来着色测试点，
            但仍会重新预测一次计算准确率（不影响使用）。
    qsvm : QSVM
        已训练好的 QSVM 模型，需实现 .predict(X) 接口。
    grid_step : float
        决策区域网格步长，越小背景越平滑但越耗时。
    margin : float
        决策边界相对于数据范围的额外边距。
    title : str | None
        图标题，默认 "QSVM Classification".
    """
    
    # 解包数据
    if not isinstance(dataset_list, (list, tuple)) or len(dataset_list) != 4:
        raise ValueError("dataset_list 必须是 (X_train, y_train, X_test, y_test_or_pred) 这样的四元组。")
    
    X_train, y_train, X_test, y_pred = dataset_list
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_test  = np.asarray(X_test)
    y_pred = np.asarray(y_pred).ravel()
    
    # 拼在一起求整体范围
    X_all = np.vstack((X_train, X_test))
    x_min, x_max = X_all[:, 0].min(), X_all[:, 0].max()
    y_min, y_max = X_all[:, 1].min(), X_all[:, 1].max()
    
    margin = 0.2
    grid_step = 0.1
    
    # 生成网格
    xx, yy = np.meshgrid(
        np.arange(x_min - margin, x_max + margin, grid_step),
        np.arange(y_min - margin, y_max + margin, grid_step),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 决策区域预测
    Z = qsvm.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(7, 6))
    
    # 背景决策区域
    plt.pcolormesh(xx, yy, Z, cmap="RdBu", shading="auto", alpha=0.6)
    
    # ===== 训练集点（实心 + 空心） =====
    # 类 0 训练
    plt.scatter(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        marker="s",
        facecolors="w",
        edgecolors="r",
        label="Class 0 (train)",
        alpha=0.8,
    )
    # 类 1 训练
    plt.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        marker="o",
        facecolors="w",
        edgecolors="b",
        label="Class 1 (train)",
        alpha=0.8,
    )
    
    # ===== 测试集点（填充色看预测类别） =====
    # 类 0 测试 (预测为 0)
    plt.scatter(
        X_test[y_pred == 0, 0],
        X_test[y_pred == 0, 1],
        marker="s",
        facecolors="r",
        edgecolors="r",
        label="Class 0 (test pred)",
        alpha=0.9,
    )
    # 类 1 测试 (预测为 1)
    plt.scatter(
        X_test[y_pred == 1, 0],
        X_test[y_pred == 1, 1],
        marker="o",
        facecolors="b",
        edgecolors="b",
        label="Class 1 (test pred)",
        alpha=0.9,
    )
    
    plt.title("QSVM Classification")
    plt.xlabel("Feature 1 / Principal Component 1")
    plt.ylabel("Feature 2 / Principal Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
