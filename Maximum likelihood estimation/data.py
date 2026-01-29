import numpy as np

# Bernoulli / Coin Toss
def generate_bernoulli_data(p, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.binomial(1, p, n)

def log_likelihood_bernoulli(p, data):
    return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))


# Normal Distribution
def generate_normal_data(mu, sigma, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mu, sigma, n)
