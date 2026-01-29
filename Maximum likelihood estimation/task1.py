import numpy as np
from data import generate_bernoulli_data
from visual import plot_mle_convergence

np.random.seed(0)

true_p = 0.7
sample_sizes = [10, 30, 50, 100, 500, 1000]
p_mle_values = []

for n in sample_sizes:
    data = generate_bernoulli_data(true_p, n)
    p_mle_values.append(data.mean())

plot_mle_convergence(
    sample_sizes,
    p_mle_values,
    true_p,
    ylabel="MLE of p",
    title="MLE Convergence with Increasing Data (Bernoulli)"
)
