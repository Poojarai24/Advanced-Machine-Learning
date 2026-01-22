# task2.py
from sklearn.svm import SVC
from data import generate_overlapping_data
from visual import plot_svm_with_margins

def main():

    X, y = generate_overlapping_data()

    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    for C in C_values:
        print(f"\nTraining SVM with C = {C}")

        svm = SVC(kernel="linear", C=C)
        svm.fit(X, y)

        print("Number of support vectors:",
              len(svm.support_vectors_))

        plot_svm_with_margins(
            X,
            y,
            svm,
            title=f"Soft Margin SVM (C = {C})"
        )


if __name__ == "__main__":
    main()
