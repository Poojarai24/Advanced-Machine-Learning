# Support Vector Machine (SVM) - Margin and Regularization

## Overview
This repository demonstrates the working principles of Support Vector Machines (SVM) using modular Python scripts.
The main focus is on understanding margins, support vectors, and the effect of the regularization parameter C.

The project avoids Jupyter notebooks and uses clean .py files for better modularity and learning clarity.

---

## Learning Objectives
After completing this project, you will understand:

- What a margin is in Support Vector Machines
- Why support vectors are important
- Difference between hard margin and soft margin SVM
- Behavior of SVM on linearly separable and overlapping data
- Effect of the regularization parameter C

---

## Project Structure

```
.
├── data.py
├── visual.py
│
├── task0.py
├── task1.py
├── task2.py
├── task3.py
│
└── README.md
```

---

## File Description

### data.py
This file handles dataset generation used across all tasks.

Functions included:
- generate_linear_separable_data()
- generate_overlapping_data()

---

### visual.py
This file contains all visualization logic.

It is responsible for plotting:
- Data points
- Decision boundary
- Margin lines (+1 and -1)
- Support vectors
- Margin shaded region

---

### task0.py - Hard Margin SVM (Linearly Separable Data)
This script demonstrates hard margin SVM on perfectly linearly separable data.

What this task shows:
- Maximum margin separation
- Very few support vectors
- No misclassification

Run using:
```
python task0.py
```

---

### task1.py - Hard Margin SVM (Overlapping Data)
This script applies hard margin SVM to overlapping data.

What this task shows:
- Increase in number of support vectors
- Sensitivity to overlapping data
- Poor generalization

Run using:
```
python task1.py
```

---

### task2.py - Soft Margin SVM (C = 1.0)
This script introduces soft margin SVM with a moderate value of C.

What this task shows:
- Allows some misclassification
- Produces a balanced margin
- Handles overlapping data better than hard margin

Run using:
```
python task2.py
```

---

### task3.py - Effect of Multiple C Values
This script demonstrates the effect of different values of C on SVM behavior.

C values used:
- 0.01
- 0.1
- 1.0
- 10.0
- 100.0

Observations:
- Smaller C gives wider margin and more support vectors
- Larger C gives narrower margin and fewer misclassifications
- Very large C behaves similar to hard margin SVM

Run using:
```
python task3.py
```

---

## Understanding the Parameter C

C is the regularization parameter in SVM.

- Small C:
  - Wider margin
  - More tolerance to misclassification

- Large C:
  - Narrower margin
  - Less tolerance to misclassification

In simple terms:
- Small C focuses on margin maximization
- Large C focuses on classification accuracy

---

## How to Run the Project

Install dependencies:
```
pip install numpy matplotlib scikit-learn
```

Run any task:
```
python task0.py
python task1.py
python task2.py
python task3.py
```

---

## Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn

---

## Educational Purpose
This project is intended for:
- Machine learning beginners
- Academic assignments
- Conceptual understanding of SVM
- Interview and viva preparation

---

## Conclusion
This project provides a clear and visual understanding of how SVM margins work and how the parameter C controls the trade-off between margin width and classification accuracy.
