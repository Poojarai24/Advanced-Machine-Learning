# task0.py
from sklearn.svm import SVC
from data import (
    generate_linear_separable_data,
    generate_overlapping_data
)
from visual import plot_data, plot_svm_with_margins

def main():
    # Part 3: Soft Margin SVM
    X_overlap, y_overlap = generate_overlapping_data()
    plot_data(X_overlap, y_overlap, "Overlapping Data")
    
    svm_soft = SVC(kernel="linear", C=1.0)
    svm_soft.fit(X_overlap, y_overlap)

    print("\nSoft Margin SVM")
    print("Number of support vectors:",
          len(svm_soft.support_vectors_))

    plot_svm_with_margins(
        X_overlap,
        y_overlap,
        svm_soft,
        title="Soft Margin SVM (C=1.0)"
    )


if __name__ == "__main__":
    main()
