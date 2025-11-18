import numpy as np
from sklearn.svm import SVC
from qiskit.quantum_info import Statevector

from ZZFeatureMap import ZZFeatureMap, ZZFeatureValue


class QSVM:
    def __init__(self, C=1.0, reps=1):
        """
        QSVM
        
        参数:
        C: 正则化参数
        """
        self.C = C
        self.reps = reps
        self.svm = None
        self.X_train = None
        self._train_states = None  # 缓存训练集态矢量，避免重复计算
        
    def _encode(self, X):
        """把一批样本提前编码成态矢量数组（加速计算）"""
        states = []
        for x in X:
            fmap = ZZFeatureMap(len(x), reps=self.reps)
            fmap.construct_circuit(x)
            sv = Statevector.from_instruction(fmap.circuit)
            states.append(sv.data)  # 直接存 numpy 向量
        return np.array(states)  # shape = (n_samples, 2**n_qubits)

    def _fidelity_kernel_from_states(self, states1, states2):
        # 计算 |<ψ_i | ψ_j>|^2
        K = np.abs(states1 @ states2.conj().T) ** 2
        return K
        
    # def quantum_kernel(self, X1, X2):
    #     """量子核函数，基于 ZZFeatureMap"""
    #     if X1.ndim == 1:
    #         X1 = X1.reshape(1, -1)
    #     if X2.ndim == 1:
    #         X2 = X2.reshape(1, -1)
            
    #     n1 = X1.shape[0]
    #     n2 = X2.shape[0]
    #     K = np.zeros((n1, n2))
    #     cnt = 0
        
    #     for i in range(n1):
    #         for j in range(n2):
    #             K[i, j] = ZZFeatureValue(X1[i], X2[j])
    #             cnt+=1
    #             if cnt % 100 == 0:
    #                 print(f"{cnt}/{n1*n2}")
    #     return K

    def fit(self, X, y):
        """训练SVM模型"""
        self.X_train = X.copy()
    
        self._train_states = self._encode(X)
        
        K = self._fidelity_kernel_from_states(self._train_states, self._train_states)
        
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(K, y)
        
        return self
    
    def predict(self, X):
        if self.svm is None:
            raise ValueError("模型尚未训练")
        test_states = self._encode(X)
        K_test = self._fidelity_kernel_from_states(test_states, self._train_states)
        return self.svm.predict(K_test)