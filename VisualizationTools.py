# VisualizationTools.py
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
    dataset_list : (X_train, y_train, X_test, y_pred)
        - X_train: 训练集特征，形状 (n_train, 2)
        - y_train: 训练集真实标签
        - X_test:  测试集特征，形状 (n_test, 2)
        - y_pred:  测试集预测标签
    qsvm : QSVM
        已训练好的 QSVM 模型，需实现 .predict(X) 接口。
    """
    
    # 解包数据
    if not isinstance(dataset_list, (list, tuple)) or len(dataset_list) != 4:
        raise ValueError("dataset_list 必须是 (X_train, y_train, X_test, y_pred) 这样的四元组。")
    
    X_train, y_train, X_test, y_pred = dataset_list
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_test  = np.asarray(X_test)
    y_pred = np.asarray(y_pred).ravel()
    
    if X_train.shape[1] != 2 or X_test.shape[1] != 2:
        raise ValueError("当前只支持二维特征的可视化，请保证 n_features=2。")
    
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
    plt.pcolormesh(xx, yy, Z, cmap="tab10", shading="auto", alpha=0.6)
    
    # ===== 按类别循环画训练/测试点 =====
    classes = np.unique(np.concatenate([y_train, y_pred]))
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'orange', 'brown', 'purple']
    markers_train = ['o', 's', '^', 'v', '<', '>', 'P', 'X', 'D', '*']
    markers_test  = ['o', 's', '^', 'v', '<', '>', 'P', 'X', 'D', '*']
    
    for idx, cls in enumerate(classes):
        c = colors[idx % len(colors)]
        mt = markers_train[idx % len(markers_train)]
        mse = markers_test[idx % len(markers_test)]
        
        # 训练集
        mask_tr = (y_train == cls)
        plt.scatter(
            X_train[mask_tr, 0],
            X_train[mask_tr, 1],
            marker=mt,
            facecolors='w',
            edgecolors=c,
            label=f"Class {cls} (train)",
            alpha=0.8,
        )
        
        # 测试集（按预测类别着色）
        mask_te = (y_pred == cls)
        plt.scatter(
            X_test[mask_te, 0],
            X_test[mask_te, 1],
            marker=mse,
            facecolors=c,
            edgecolors=c,
            label=f"Class {cls} (test pred)",
            alpha=0.9,
        )
    
    plt.title("QSVM Classification (multi-class)")
    plt.xlabel("Feature 1 / Principal Component 1")
    plt.ylabel("Feature 2 / Principal Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
