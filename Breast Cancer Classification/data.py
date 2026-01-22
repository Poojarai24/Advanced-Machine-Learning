import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

def load_data(test_size=0.2, random_state=42):
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA for visualization (2D)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, y_train, y_test

def generate_large_dataset(X, y, n_samples=5000, noise=0.01):
    """
    Expands dataset using bootstrapping + Gaussian noise
    """
    X_resampled, y_resampled = resample(
        X, y,
        n_samples=n_samples,
        stratify=y,
        random_state=42
    )

    # Add small noise to avoid exact duplicates
    X_resampled = X_resampled + noise * np.random.randn(*X_resampled.shape)

    return X_resampled, y_resampled