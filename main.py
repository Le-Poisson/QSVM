import numpy as np

from PreDataset import PreDataset
from QSVM import QSVM
from sklearn.metrics import accuracy_score, classification_report
from VisualizationTools import plot_QSVM_decision_boundary

if __name__ == "__main__":

    print("加载数据集...")
    
    X_train, y_train, X_test, y_test = PreDataset(data_id=3, n_features=2, max_samples=100, train_ratio=0.85)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    print("训练QSVM模型...")
    
    qsvm = QSVM(C=1.0, reps=1)
    qsvm.fit(X_train, y_train)
    y_pred = qsvm.predict(X_test)
    
    accuracy_custom = accuracy_score(y_test, y_pred)
    print(f"QSVM准确率: {accuracy_custom:.4f}")
    
    classes = np.unique(np.concatenate([y_train, y_test]))
    target_names = [f"Class {int(c)}" for c in classes]
    
    print("\nQSVM分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 只在二维特征时画决策边界
    if X_train.shape[1] == 2:
        plot_QSVM_decision_boundary([X_train, y_train, X_test, y_pred], qsvm)