# Breast Cancer Classification – Linear vs Polynomial vs RBF Models

This project demonstrates how different machine learning models behave on the **Breast Cancer Wisconsin dataset** when trained on:

- The original dataset (569 samples, 30 features)
- An expanded (large) dataset generated using bootstrapping and noise

The goal is to **analyze model behavior, accuracy, scalability, and decision boundaries** using visualization.

---

## 📌 Dataset Used

- **Dataset**: Breast Cancer Wisconsin Diagnostic Dataset  
- **Source**: `sklearn.datasets.load_breast_cancer`
- **Samples**: 569
- **Features**: 30 (numeric)
- **Classes**:
  - `0` → Malignant
  - `1` → Benign

---

## Project Structure

```
.
│
├── data.py # Data loading, preprocessing, PCA, dataset expansion
├── visual.py # Decision boundary visualization utilities
├── task0.py # Training & visualization on original dataset
├── task1.py # Training & evaluation on expanded (large) dataset
└── README.md
```

---

## 🔹 File Descriptions

### `data.py`
Handles all data-related operations:
- Loading the breast cancer dataset
- Train-test splitting with stratification
- Feature scaling using StandardScaler
- PCA (2D) transformation for visualization
- Dataset expansion using bootstrapping + Gaussian noise

---

### `visual.py`
Contains visualization utilities:
- Plots 2D decision boundaries
- Uses PCA-reduced features
- Optionally displays model accuracy in plot titles
- No model training or evaluation logic (single responsibility)

---

### `task0.py`
Main experiment on the **original dataset**:
- Trains and evaluates:
  - Linear Logistic Regression
  - Polynomial Logistic Regression (Degree = 2)
  - SVM with RBF Kernel
- Computes **test accuracy**
- Visualizes decision boundaries for each model
- Prints a comparative accuracy summary

---

### `task1.py`
Large dataset experiment:
- Expands training data (e.g., 569 → 10,000 samples)
- Uses bootstrapping with noise injection
- Trains the same three models
- Evaluates model performance on the original test set
- Analyzes scalability and robustness

---

## 🔹 Models Implemented

1. **Linear Logistic Regression**
   - Linear decision boundary
   - Low variance, stable performance

2. **Polynomial Logistic Regression (Degree = 2)**
   - Non-linear decision boundary via feature expansion
   - Captures feature interactions

3. **Support Vector Machine (RBF Kernel)**
   - Highly flexible non-linear model
   - Sensitive to data density and noise

---

## 📊 Visualization Approach

- Since the dataset has 30 features, **PCA is used to reduce features to 2D**
- Decision boundaries are plotted in PCA space
- This allows visual comparison of:
  - Linear vs non-linear behavior
  - Model flexibility
  - Overfitting tendencies

---

## 📈 Evaluation Metric

- **Accuracy** (computed on the test set)
- Accuracy is calculated in `task0.py` and `task1.py`
- Visualization functions only *display* accuracy, not compute it

---

## 🔹 Dataset Expansion Strategy

To simulate large datasets:
- Bootstrapped resampling is applied
- Small Gaussian noise is added to avoid exact duplicates
- Class distribution is preserved using stratified sampling

⚠️ Note:
> This does **not** add new information.  
> It evaluates **scalability and stability**, not true generalization.

---

## ▶️ How to Run

### Run original dataset experiment
```bash
python task0.py
