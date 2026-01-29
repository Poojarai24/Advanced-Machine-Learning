import numpy as np
from data import generate_bernoulli_data, log_likelihood_bernoulli
from visual import plot_log_likelihood

np.random.seed(1)

true_p = 0.7
data = generate_bernoulli_data(true_p, 10)

print("Observed coin tosses (1=Head, 0=Tail):")
print(data)
print("Number of heads:", data.sum())

# Likelihood
p_values = np.linspace(0.01, 0.99, 100)
ll_values = [log_likelihood_bernoulli(p, data) for p in p_values]

plot_log_likelihood(p_values, ll_values)

# MLE
p_mle = data.mean()
print("MLE estimate of p:", p_mle)

p_mle_grid = p_values[np.argmax(ll_values)]
print("MLE from likelihood curve:", p_mle_grid)
