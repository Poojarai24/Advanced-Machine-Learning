# 📈 Maximum Likelihood Estimation (MLE): Bernoulli & Normal Distributions
 
This mini-project demonstrates the concept of **Maximum Likelihood Estimation (MLE)** using simple and intuitive examples:
 
- 🎲 **Bernoulli distribution (coin toss)**
- 🔔 **Normal distribution (Gaussian mean estimation)**
 
Through simulations and visualizations, the project shows how **MLE works**, how **log-likelihood behaves**, and how **MLE estimates converge to true parameters** as data increases.
 
---
 
## 🎯 Objectives
 
- Understand **Maximum Likelihood Estimation**
- Visualize **log-likelihood functions**
- Compute MLE for:
  - Bernoulli probability parameter `p`
  - Normal distribution mean `μ`
- Observe **MLE convergence** with increasing sample size
 
---
 
## 📂 Folder Structure
 
```bash
MLE_Estimation/
│
├── data.py          # Data generation & likelihood functions
├── visual.py        # Plotting utilities
│
├── task0.py         # Bernoulli MLE + log-likelihood visualization
├── task1.py         # MLE convergence for Bernoulli distribution
├── task2.py         # MLE convergence for Normal distribution (mean)
│
└── README.md        # Documentation
```
 
---
 
## 📊 Concepts Covered
 
- Maximum Likelihood Estimation (MLE)
- Bernoulli distribution
- Binomial sampling (coin toss)
- Log-likelihood function
- Normal (Gaussian) distribution
- Consistency of MLE
- Effect of sample size on estimation
 
---
 
## 🧪 Tasks Explanation
 
---
 
### ✅ Task 0: Bernoulli MLE & Log-Likelihood
 
📌 `task0.py`
 
This task simulates a **coin toss experiment**.
 
Steps:
- Generate Bernoulli data with true probability `p = 0.7`
- Compute log-likelihood over a range of `p` values
- Plot **Log-Likelihood vs p**
- Estimate `p` using:
  - Sample mean (analytical MLE)
  - Maximum of likelihood curve (grid search)
 
✅ Key Learning:
 
> The MLE of a Bernoulli distribution is simply the **sample mean**.
 
Run:
```bash
python task0.py
```
 
---
 
### ✅ Task 1: MLE Convergence (Bernoulli)
 
📌 `task1.py`
 
This task shows how the **MLE of p converges** as sample size increases.
 
Steps:
- Generate Bernoulli data for increasing sample sizes
- Compute MLE (`p̂`) for each size
- Plot MLE estimate vs sample size
 
✅ Key Learning:
 
> As the number of samples increases, the MLE converges to the true parameter value.
 
Run:
```bash
python task1.py
```
 
---
 
### ✅ Task 2: MLE Convergence (Normal Distribution)
 
📌 `task2.py`
 
This task demonstrates MLE for a **Normal distribution mean**.
 
Steps:
- Generate Normal data with known `μ` and `σ`
- Estimate the mean using sample average
- Plot convergence of mean estimate as sample size increases
 
✅ Key Learning:
 
> The MLE of the mean of a Normal distribution is the **sample mean**, and it is a consistent estimator.
 
Run:
```bash
python task2.py
```
 
---
 
## 📈 Visualizations Included
 
- Log-Likelihood vs Parameter
- MLE estimate vs Sample Size
- True parameter value reference line
 
These plots provide **intuition behind likelihood maximization and convergence**.
 
---
 
## ▶️ How to Run
 
### ✅ Install dependencies
 
```bash
pip install numpy matplotlib
```
 
### ✅ Run tasks
 
```bash
python task0.py
python task1.py
python task2.py
```
 
Each script produces plots and prints relevant outputs.
 
---
 
## ⚠️ Notes
 
- Random seeds are fixed for reproducibility
- Only parameter estimation is demonstrated (no hypothesis testing)
- Designed for learning and intuition, not production use
 
---
 
## 🏁 Conclusion
 
This project clearly illustrates:
 
- How Maximum Likelihood Estimation works
- Why log-likelihood is used instead of likelihood
- How increasing data improves parameter estimation
- Practical MLE examples for common distributions
 
It serves as a **strong foundation for statistics, machine learning, and probabilistic modeling**.
 
---