📐 Support Vector Machine (SVM): Margin & Regularization (C)
📌 Project Overview

This repository is an educational implementation of Support Vector Machines (SVMs) focused on understanding:

Maximum margin principle

Support vectors

Hard margin vs soft margin SVM

Effect of the regularization parameter C

Behavior of SVM on linearly separable and overlapping datasets

The project is implemented using modular Python scripts instead of a single notebook, making the learning flow clear and structured.

🎯 Learning Goals

This project helps a learner to:

Visually understand SVM decision boundaries

See how margins are formed

Observe how support vectors change

Understand the bias–variance trade-off controlled by parameter C

Compare SVM behavior across multiple experiments

🗂️ Project Structure
.
├── data.py        # Dataset generation
├── visual.py      # Plotting & visualization
│
├── task0.py       # Hard Margin SVM (Linearly Separable Data)
├── task1.py       # Hard Margin SVM (Overlapping Data)
├── task2.py       # Soft Margin SVM (C = 1.0)
├── task3.py       # Effect of Different C Values
│
└── README.md

📁 File Descriptions
🔹 data.py

Responsible for generating datasets used across all experiments.

Linearly separable data

Overlapping (non-linearly separable) data

This ensures data generation is isolated from model logic.

🔹 visual.py

Contains all visualization logic, including:

Scatter plots of data

SVM decision boundary

Margin lines (+1 and −1)

Highlighting support vectors

Shaded margin region

Keeping visualization separate improves readability and reuse.

🧪 Experiments (Task Files)

Each task file represents one focused experiment.

🧩 task0.py — Hard Margin SVM (Linearly Separable Data)

Uses perfectly separable data

Trains SVM with a very large C (C = 1e6)

Demonstrates:

Maximum margin classifier

Minimal number of support vectors

Clean decision boundary

📌 Key takeaway:
Hard Margin SVM works well only when data is perfectly separable.

🧩 task1.py — Hard Margin SVM (Overlapping Data)

Uses overlapping data

Applies hard margin SVM on non-separable data

Shows:

Increase in number of support vectors

Sensitivity to noise

Poor generalization

📌 Key takeaway:
Hard margin SVM is not suitable for noisy or overlapping data.

🧩 task2.py — Soft Margin SVM (Fixed C = 1.0)

Uses overlapping data

Trains soft margin SVM (C = 1.0)

Allows margin violations

Produces better generalization

📌 Key takeaway:
Soft margin SVM balances margin size and classification error.

🧩 task3.py — Effect of Regularization Parameter C

Trains SVM with multiple C values:

C = [0.01, 0.1, 1.0, 10.0, 100.0]


Visualizes how decision boundary and margin change

Prints number of support vectors for each C

📌 Key observations:

C Value	Margin	Misclassification	Support Vectors
Small C	Wide	High	More
Large C	Narrow	Low	Fewer
🔍 Understanding the Parameter C

C controls the trade-off between margin width and classification accuracy.

Small C → Larger margin, allows misclassification (high bias)

Large C → Smaller margin, strict classification (low bias)

Very large C → Approaches hard margin behavior

▶️ How to Run
Install Dependencies
pip install numpy matplotlib scikit-learn

Run Experiments
python task0.py   # Hard margin on separable data
python task1.py   # Hard margin on overlapping data
python task2.py   # Soft margin (C = 1.0)
python task3.py   # Multiple C value comparison

🛠️ Technologies Used

Python

NumPy

Matplotlib

Scikit-learn

🎓 Intended Audience

This project is suitable for:

Machine Learning beginners

Academic assignments

Conceptual demonstrations

Interview and viva preparation

✅ Summary

This project demonstrates how SVM margins and regularization affect model behavior.
By breaking the learning into multiple focused tasks, it provides a clear, step-by-step understanding of SVM concepts through visualization.