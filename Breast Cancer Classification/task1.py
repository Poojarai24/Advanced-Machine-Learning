from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data import load_data, generate_large_dataset

# ----------------------------------
# 1️⃣ Load original data
# ----------------------------------
X_train_scaled, X_test_scaled, _, _, y_train, y_test = load_data()

print("Original training size:", X_train_scaled.shape)

# ----------------------------------
# 2️⃣ Expand dataset
# ----------------------------------
X_large, y_large = generate_large_dataset(
    X_train_scaled,
    y_train,
    n_samples=10000,
    noise=0.02
)

print("Expanded training size:", X_large.shape)

# ----------------------------------
# 3️⃣ PCA for visualization / consistency
# ----------------------------------
pca = PCA(n_components=2)
X_large_pca = pca.fit_transform(X_large)
X_test_pca = pca.transform(X_test_scaled)

# ----------------------------------
# 4️⃣ Models
# ----------------------------------
models = {
    "Linear": LogisticRegression(),
    "Poly_deg2": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "RBF": SVC(kernel="rbf", gamma="scale")
}

# ----------------------------------
# 5️⃣ Train & Evaluate
# ----------------------------------
print("\n===== Accuracy on Expanded Dataset =====")

for name, model in models.items():
    model.fit(X_large_pca, y_large)
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc:.4f}")
