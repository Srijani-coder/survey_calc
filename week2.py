import math
from week1 import get_sample_values, compute_sample_statistic

def population_mean(population_values):
    """
    Concept: Population Mean
    Calculates the average value of a variable across the entire population.

    Parameters:
    - population_values (list): A list of values representing the population.
    """
    return sum(population_values) / len(population_values)

def population_variance(population_values):
    """
    Concept: Population Variance
    Measures the spread of values in the population from the population mean.

    Parameters:
    - population_values (list): A list of values representing the population.
    """
    mu = population_mean(population_values)
    return sum((x - mu) ** 2 for x in population_values) / (len(population_values) - 1)

def sample_mean(sample_values):
    """
    Concept: Sample Mean
    Computes the average value from a sample.

    Parameters:
    - sample_values (list): A list of sample data values.
    """
    return sum(sample_values) / len(sample_values)

def sample_variance(sample_values):
    """
    Concept: Sample Variance
    Estimates variance in the population based on a sample.

    Parameters:
    - sample_values (list): A list of sample data values.
    """
    y_bar = sample_mean(sample_values)
    return sum((y - y_bar) ** 2 for y in sample_values) / (len(sample_values) - 1)

def expected_value(x_values, p_values):
    """
    Concept: Expected Value
    Computes the weighted average (expectation) of a discrete random variable.

    Parameters:
    - x_values (list): Possible values of the variable.
    - p_values (list): Corresponding probabilities.
    """
    return sum(p * x for x, p in zip(x_values, p_values))

def expected_value_function(x_values, p_values, g):
    """
    Concept: Expected Value of a Function
    Computes E[g(X)] for a function g over a discrete distribution.

    Parameters:
    - x_values (list): Possible values of the variable.
    - p_values (list): Corresponding probabilities.
    - g (function): A function applied to each x_value.
    """
    return sum(p * g(x) for x, p in zip(x_values, p_values))

def variance_discrete(x_values, p_values):
    """
    Concept: Variance of Discrete Random Variable
    Measures spread of a discrete variable around its expected value.

    Parameters:
    - x_values (list): Possible values of the variable.
    - p_values (list): Corresponding probabilities.
    """
    mean = expected_value(x_values, p_values)
    return sum(p * (x - mean) ** 2 for x, p in zip(x_values, p_values))

def var_sample_mean_fpc(pop_variance, n, N):
    """
    Concept: Finite Population Correction (FPC)
    Adjusts variance of sample mean when sampling without replacement from finite population.

    Parameters:
    - pop_variance (float): Variance of the population.
    - n (int): Sample size.
    - N (int): Population size.
    """
    f = n / N
    return (pop_variance / n) * (1 - f)

def ci_mean_known_variance(y_bar, z, sigma, n, N):
    """
    Concept: Confidence Interval for Mean (Known Variance)
    Computes the CI around the sample mean using known population variance and FPC.

    Parameters:
    - y_bar (float): Sample mean.
    - z (float): Z-value for desired confidence level.
    - sigma (float): Known population standard deviation.
    - n (int): Sample size.
    - N (int): Population size.
    """
    f = n / N
    margin = z * sigma * math.sqrt((1 - f) / n)
    return y_bar - margin, y_bar + margin

def required_sample_size(z, sigma, e, N):
    """
    Concept: Required Sample Size
    Calculates required sample size to achieve desired margin of error.

    Parameters:
    - z (float): Z-value for desired confidence level.
    - sigma (float): Standard deviation.
    - e (float): Desired margin of error.
    - N (int): Population size.
    """
    numerator = (z * sigma) ** 2
    return int(math.ceil(numerator / e ** 2 + numerator / (e ** 2 * N)))

if __name__ == "__main__":
    population = [1, 3, 5]
    sample = [1, 3]
    probs = [0.3, 0.4, 0.3]

    print("Population Mean:", population_mean(population))
    print("Population Variance:", population_variance(population))
    print("Sample Mean:", sample_mean(sample))
    print("Sample Variance:", sample_variance(sample))
    print("Expected Value of X:", expected_value(population, probs))
    print("Expected Value of X^2:", expected_value_function(population, probs, lambda x: x ** 2))
    print("Variance of Discrete X:", variance_discrete(population, probs))
    print("Variance of Sample Mean with FPC:", var_sample_mean_fpc(2.25, 2, 3))
    print("95% CI (σ²=2.25):", ci_mean_known_variance(3.0, 1.96, math.sqrt(2.25), 2, 3))
    print("Required Sample Size (e=0.5):", required_sample_size(1.96, 2.25, 0.5, 1000))
