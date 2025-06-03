import statistics, math
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import statistics

def survey_mean(value_list):
    """
    Concept: Survey Mean
    Computes the sample mean and standard error of a list of numeric survey values.

    Parameters:
    - value_list (list): Numeric values from survey responses.
    """
    mean = statistics.mean(value_list)
    se = statistics.stdev(value_list) / math.sqrt(len(value_list))
    return mean, se

def survey_median(value_list):
    """
    Concept: Survey Median
    Computes the median and its approximate standard error from survey values.

    Parameters:
    - value_list (list): Numeric survey responses.
    """
    median = statistics.median(value_list)
    se = 1.2533 * statistics.stdev(value_list) / math.sqrt(len(value_list))
    return median, se

def survey_quantile(value_list, quantiles=[0.25, 0.75]):
    """
    Concept: Survey Quantiles
    Computes selected quantiles and estimates standard errors for each.

    Parameters:
    - value_list (list): Numeric survey responses.
    - quantiles (list): List of quantile probabilities (e.g., [0.25, 0.75]).
    """
    q_values = np.quantile(value_list, quantiles)
    se = [1.2533 * np.std(value_list, ddof=1) / np.sqrt(len(value_list))] * len(quantiles)
    return list(zip(quantiles, q_values, se))

def survey_ratio(numerator, denominator):
    """
    Concept: Survey Ratio Estimate
    Estimates the ratio of two variables (Y/X) and its standard error.

    Parameters:
    - numerator (list): List of Y-values (e.g., income).
    - denominator (list): List of X-values (e.g., household size).
    """
    ratio = np.sum(numerator) / np.sum(denominator)
    se = np.std(np.array(numerator) / np.array(denominator), ddof=1) / np.sqrt(len(numerator))
    return ratio, se

def design_effect(variance_design, variance_srs):
    """
    Concept: Design Effect (Deff)
    Quantifies how much the sampling design increases variance relative to simple random sampling.

    Parameters:
    - variance_design (float): Observed variance from complex survey design.
    - variance_srs (float): Expected variance under simple random sampling.
    """
    return variance_design / variance_srs

def t_test_1sample(sample, population_mean=0):
    """
    Concept: One-Sample T-Test
    Tests whether the sample mean differs from a known population mean.

    Parameters:
    - sample (list): Sample data.
    - population_mean (float): Population mean to compare against (default = 0).
    """
    return stats.ttest_1samp(sample, population_mean, nan_policy='omit')

def t_test_2sample(group1, group2):
    """
    Concept: Two-Sample T-Test
    Tests if two independent groups differ significantly in means.

    Parameters:
    - group1 (list): First sample group.
    - group2 (list): Second sample group.
    """
    return stats.ttest_ind(group1, group2, nan_policy='omit')

def regression_coefficients(x, y):
    """
    Concept: Linear Regression Coefficients
    Fits simple linear regression and returns intercept and slope.

    Parameters:
    - x (list): Predictor variable values.
    - y (list): Response variable values.
    """
    model = LinearRegression().fit(np.array(x).reshape(-1, 1), y)
    return model.intercept_, model.coef_[0]

def domain_analysis_mean(groups, values):
    """
    Concept: Domain-Level Mean
    Computes mean values for each domain or group in the survey.

    Parameters:
    - groups (list): Group labels (e.g., gender, region).
    - values (list): Corresponding numeric values.
    """
    from collections import defaultdict
    import statistics
    group_means = {}
    group_data = defaultdict(list)
    for g, v in zip(groups, values):
        group_data[g].append(v)
    for g, v in group_data.items():
        group_means[g] = statistics.mean(v)
    return group_means

def domain_analysis_total(groups, values):
    """
    Concept: Domain-Level Total
    Computes total value for each domain or group.

    Parameters:
    - groups (list): Group identifiers.
    - values (list): Numeric values associated with each group.
    """
    from collections import defaultdict
    group_totals = {}
    group_data = defaultdict(list)
    for g, v in zip(groups, values):
        group_data[g].append(v)
    for g, v in group_data.items():
        group_totals[g] = sum(v)
    return group_totals
