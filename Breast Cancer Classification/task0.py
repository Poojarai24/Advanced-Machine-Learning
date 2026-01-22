from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data import load_data
from visual import plot_decision_boundary

# Load data
X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, y_train, y_test = load_data()

# 1️⃣ Linear Logistic Regression
linear_model = LogisticRegression()
linear_model.fit(X_train_pca, y_train)

y_pred_linear = linear_model.predict(X_test_pca)
acc_linear = accuracy_score(y_test, y_pred_linear)

print(f"Linear Model Accuracy: {acc_linear:.4f}")

plot_decision_boundary(
    linear_model,
    X_train_pca,
    y_train,
    "Linear Logistic Regression (PCA)",
    acc_linear
)

# 2️⃣ Polynomial Degree = 2
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("clf", LogisticRegression(max_iter=1000))
])

poly_model.fit(X_train_pca, y_train)

y_pred_poly = poly_model.predict(X_test_pca)
acc_poly = accuracy_score(y_test, y_pred_poly)

print(f"Polynomial Degree-2 Accuracy: {acc_poly:.4f}")

plot_decision_boundary(
    poly_model,
    X_train_pca,
    y_train,
    "Polynomial Logistic Regression (Degree=2)",
    acc_poly
)

# 3️⃣ SVM with RBF Kernel
rbf_model = SVC(kernel="rbf", gamma="scale")
rbf_model.fit(X_train_pca, y_train)

y_pred_rbf = rbf_model.predict(X_test_pca)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"RBF SVM Accuracy: {acc_rbf:.4f}")

plot_decision_boundary(
    rbf_model,
    X_train_pca,
    y_train,
    "SVM with RBF Kernel",
    acc_rbf
)

# Final Summary
print("\n===== Model Accuracy Summary =====")
print(f"Linear        : {acc_linear:.4f}")
print(f"Polynomial d=2: {acc_poly:.4f}")
print(f"RBF SVM       : {acc_rbf:.4f}")
