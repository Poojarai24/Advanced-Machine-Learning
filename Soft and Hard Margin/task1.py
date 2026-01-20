# task0.py
from sklearn.svm import SVC
from data import generate_overlapping_data

from visual import plot_data, plot_svm_with_margins

def main():  
    # Part 2: Overlapping Data
    X_overlap, y_overlap = generate_overlapping_data()
    plot_data(X_overlap, y_overlap, "Overlapping Data")

    # Hard margin on overlapping data
    svm_hard_overlap = SVC(kernel="linear", C=1e6)
    svm_hard_overlap.fit(X_overlap, y_overlap)

    print("\nHard Margin on Overlapping Data")
    print("Number of support vectors:",
          len(svm_hard_overlap.support_vectors_))

    plot_svm_with_margins(
        X_overlap,
        y_overlap,
        svm_hard_overlap,
        title="Hard Margin SVM on Overlapping Data"
    )


if __name__ == "__main__":
    main()
