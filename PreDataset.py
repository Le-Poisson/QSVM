import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def PreDataset(data_id, n_features=2, max_samples=100, train_ratio=0.85):
    """
    预处理数据集
    参数：
    data_id: 数据集编号
    n_features: 选择的特征数量
    max_samples: 最大样本数量
    train_ratio: 训练集比例
    返回：
    features: 特征数据
    labels: 标签数据
    """
    X = None
    y = None
    
    if data_id == 1:
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=max_samples, n_features=n_features, centers=2, random_state=15, shuffle=True)
    elif data_id == 2:
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

        X_df = breast_cancer_wisconsin_diagnostic.data.features
        y_df = breast_cancer_wisconsin_diagnostic.data.targets
        y_df = y_df.iloc[:, 0].map({'M': 1, 'B': 0})
        
        X = X_df.values
        y = y_df.values.flatten()
        
    # 先进行标准化，再进行降维
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # PCA 降维到指定维度
    pca = PCA(n_components=n_features)
    X = pca.fit_transform(X_scaled)
    
    if max_samples < X.shape[0]:
        X = X[:max_samples]
        y = y[:max_samples]
            
    # 特征数值范围限制在 [0, π] 之间
    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=int(max_samples*train_ratio), random_state=42, shuffle=True
    )
    
    return train_X, train_y, test_X, test_y