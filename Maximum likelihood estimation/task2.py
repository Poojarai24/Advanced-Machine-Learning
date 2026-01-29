from data import generate_normal_data
from visual import plot_mle_convergence
import numpy as np

np.random.seed(0)
true_mu = 5.0
true_sigma = 2.0

sample_sizes = [10, 30, 50, 100, 500, 1000]
mu_estimates = []

for n in sample_sizes:
    data = generate_normal_data(true_mu, true_sigma, n)
    mu_estimates.append(data.mean())

plot_mle_convergence(
    sample_sizes,
    mu_estimates,
    true_mu,
    ylabel="MLE of Mean",
    title="MLE Convergence of Mean with Increasing Data (Normal)"
)
