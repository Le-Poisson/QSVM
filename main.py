from PreDataset import PreDataset
from QSVM import QSVM
from sklearn.metrics import accuracy_score, classification_report
from VisualizationTools import plot_QSVM_decision_boundary

if __name__ == "__main__":

    print("加载数据集...")
    
    X_train, y_train, X_test, y_test = PreDataset(data_id=2, n_features=2, max_samples=100, train_ratio=0.85)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    print("训练QSVM模型...")
    
    qsvm = QSVM()
    qsvm.fit(X_train, y_train)
    y_pred = qsvm.predict(X_test)
    
    accuracy_custom = accuracy_score(y_test, y_pred)
    print(f"QSVM准确率: {accuracy_custom:.4f}")
    print("\nQSVM分类报告:")
    print(classification_report(y_test, y_pred, 
                          target_names=['特征 1', '特征 2']))
    
    plot_QSVM_decision_boundary([X_train, y_train, X_test, y_pred], qsvm)
    