# task0.py
from sklearn.svm import SVC
from data import generate_linear_separable_data
from visual import plot_data, plot_svm_with_margins

def main():
    
    # Part 1: Linearly Separable Data
  
    X, y = generate_linear_separable_data()
    plot_data(X, y, "Linearly Separable Data")

    svm_hard = SVC(kernel="linear", C=1e6)
    svm_hard.fit(X, y)

    print("Hard Margin SVM")
    print("Number of support vectors:", len(svm_hard.support_vectors_))
    print("w:", svm_hard.coef_[0])
    print("b:", svm_hard.intercept_[0])

    plot_svm_with_margins(
        X, y, svm_hard,
        title="Hard Margin SVM with Margin Visualization"
    )


if __name__ == "__main__":
    main()
