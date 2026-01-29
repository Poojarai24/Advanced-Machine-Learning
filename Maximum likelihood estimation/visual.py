import matplotlib.pyplot as plt

def plot_log_likelihood(p_values, ll_values):
    plt.plot(p_values, ll_values)
    plt.xlabel("p")
    plt.ylabel("Log Likelihood")
    plt.title("Log Likelihood vs p")
    plt.show()


def plot_mle_convergence(sample_sizes, estimates, true_value, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, estimates, marker='o', label="MLE Estimate")
    plt.axhline(y=true_value, linestyle='--', label="True Value")
    plt.xlabel("Sample Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
