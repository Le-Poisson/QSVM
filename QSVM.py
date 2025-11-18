import numpy as np
from sklearn.svm import SVC

class QSVM:
    def __init__(self, C=1.0):
        """
        QSVM
        
        参数:
        C: 正则化参数
        """
        self.C = C
        self.svm = None
        self.X_train = None
        
    def quantum_kernel(self, X1, X2):
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
            
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        cnt = 0
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = ZZFeatureValue(X1[i], X2[j])
                cnt+=1
                if cnt % 100 == 0:
                    print(f"{cnt}/{n1*n2}")
        return K

    def fit(self, X, y):
        """训练SVM模型"""
        self.X_train = X.copy()
        
        # 使用预计算核函数
        kernel_matrix = self.quantum_kernel(X, X)
        
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(kernel_matrix, y)
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.svm is None:
            raise ValueError("模型尚未训练")
            
        kernel_matrix = self.quantum_kernel(X, self.X_train)
        return self.svm.predict(kernel_matrix)