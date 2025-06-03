import math
from week2 import population_mean, population_variance, sample_mean, sample_variance

# Week 3: Proportion and Stratified Sampling Formulas

def population_proportion(binary_population):
    """
    Concept: Population Proportion
    Calculates the proportion of 1s in a binary population (e.g., success rate).

    Parameters:
    - binary_population (list): List of binary (0/1) values.
    """
    return sum(binary_population) / len(binary_population)

def binary_population_variance(p, N):
    """
    Concept: Population Variance for Binary Variable
    Computes variance for binary population using p(1-p) with finite population correction.

    Parameters:
    - p (float): Proportion of successes.
    - N (int): Population size.
    """
    return (N / (N - 1)) * p * (1 - p)

def sample_proportion(sample_values):
    """
    Concept: Sample Proportion
    Calculates sample proportion (mean of binary 0/1 sample).

    Parameters:
    - sample_values (list): List of binary values (0 or 1).
    """
    return sum(sample_values) / len(sample_values)

def binary_sample_variance(phat, n):
    """
    Concept: Sample Variance for Binary Data
    Computes sample variance estimate from sample proportion phat.

    Parameters:
    - phat (float): Sample proportion.
    - n (int): Sample size.
    """
    return (n / (n - 1)) * phat * (1 - phat)

def var_phat(p, n, N):
    """
    Concept: Variance of Sample Proportion (with FPC)
    Calculates variance of sample proportion with finite population correction.

    Parameters:
    - p (float): Population proportion.
    - n (int): Sample size.
    - N (int): Population size.
    """
    f = n / N
    return (1 - f) * p * (1 - p) / n

def ci_proportion(phat, z, n, N):
    """
    Concept: Confidence Interval for Proportion
    Constructs a confidence interval for the sample proportion.

    Parameters:
    - phat (float): Sample proportion.
    - z (float): Z-score for confidence level.
    - n (int): Sample size.
    - N (int): Population size.
    """
    f = n / N
    se = math.sqrt((1 - f) * phat * (1 - phat) / n)
    return phat - z * se, phat + z * se

def required_sample_size_proportion(z, p, e, N):
    """
    Concept: Required Sample Size for Proportion Estimate
    Determines sample size to achieve a desired margin of error.

    Parameters:
    - z (float): Z-score.
    - p (float): Estimated proportion.
    - e (float): Desired margin of error.
    - N (int): Population size.
    """
    numerator = (z ** 2) * p * (1 - p)
    return math.ceil(numerator / (e ** 2 + numerator / N))

# ---------- Stratified Sampling ----------

def stratified_mean(Ni, mui):
    """
    Concept: Stratified Mean
    Calculates population mean from multiple strata.

    Parameters:
    - Ni (list): Sizes of each stratum.
    - mui (list): Mean value in each stratum.
    """
    N = sum(Ni)
    return sum(Ni[i] * mui[i] for i in range(len(Ni))) / N

def stratified_variance(Ni, sigma2i, mui):
    """
    Concept: Stratified Population Variance
    Combines within-stratum and between-stratum variance.

    Parameters:
    - Ni (list): Sizes of each stratum.
    - sigma2i (list): Variances within each stratum.
    - mui (list): Means for each stratum.
    """
    N = sum(Ni)
    mu = stratified_mean(Ni, mui)
    within = sum((Ni[i] - 1) * sigma2i[i] for i in range(len(Ni)))
    between = sum(Ni[i] * (mui[i] - mu) ** 2 for i in range(len(Ni)))
    return (within + between) / (N - 1)

def stratified_sample_mean(Ni, sample_means):
    """
    Concept: Stratified Sample Mean
    Calculates overall sample mean from stratified sampling.

    Parameters:
    - Ni (list): Stratum sizes.
    - sample_means (list): Sample means in each stratum.
    """
    N = sum(Ni)
    return sum((Ni[i] / N) * sample_means[i] for i in range(len(Ni)))

def stratified_var_estimate(Ni, sigma2i, ni):
    """
    Concept: Variance of Stratified Mean (Known Variance)
    Estimates variance of stratified sample mean using known variances.

    Parameters:
    - Ni (list): Stratum sizes.
    - sigma2i (list): Known variances in each stratum.
    - ni (list): Sample sizes in each stratum.
    """
    N = sum(Ni)
    return sum(((Ni[i] / N) ** 2) * ((1 - (ni[i] / Ni[i])) * sigma2i[i] / ni[i]) for i in range(len(Ni)))

def stratified_var_estimate_sample(Ni, si2, ni):
    """
    Concept: Variance Estimate Using Sample Variances
    Estimates variance of stratified mean when only sample variances are known.

    Parameters:
    - Ni (list): Stratum sizes.
    - si2 (list): Sample variances.
    - ni (list): Sample sizes.
    """
    N = sum(Ni)
    return sum(((Ni[i] / N) ** 2) * ((1 - (ni[i] / Ni[i])) * si2[i] / ni[i]) for i in range(len(Ni)))
