import math

# Week 4: Extended Stratified Sampling Execution

def weighted_sample_mean(Ni, sample_means):
    """
    Concept: Weighted Sample Mean
    Computes the stratified estimate of the population mean.

    Parameters:
    - Ni (list): Sizes of each stratum.
    - sample_means (list): Sample means for each stratum.
    """
    N = sum(Ni)
    return sum((Ni[i] / N) * sample_means[i] for i in range(len(Ni)))

def weighted_mean_variance(Ni, sigma2i, ni):
    """
    Concept: Variance of Weighted Mean (with known variances)
    Estimates the variance of the weighted sample mean using population variances.

    Parameters:
    - Ni (list): Stratum sizes.
    - sigma2i (list): Known population variances in each stratum.
    - ni (list): Sample sizes in each stratum.
    """
    N = sum(Ni)
    return sum(((Ni[i]/N)**2) * ((1 - (ni[i] / Ni[i])) * sigma2i[i] / ni[i]) for i in range(len(Ni)))

def estimated_variance_weighted_mean(Ni, si2, ni):
    """
    Concept: Estimated Variance of Weighted Mean (using sample variances)
    Estimates variance of the stratified mean using sample variances si².

    Parameters:
    - Ni (list): Stratum sizes.
    - si2 (list): Sample variances for each stratum.
    - ni (list): Sample sizes for each stratum.
    """
    N = sum(Ni)
    return sum(((Ni[i]/N)**2) * ((1 - (ni[i] / Ni[i])) * si2[i] / ni[i]) for i in range(len(Ni)))

def ci_weighted_mean(y_str, var_estimate, z):
    """
    Concept: Confidence Interval for Weighted Mean
    Computes confidence interval for the estimated population mean.

    Parameters:
    - y_str (float): Weighted sample mean.
    - var_estimate (float): Estimated variance of the mean.
    - z (float): Z-value for the desired confidence level.
    """
    se = math.sqrt(var_estimate)
    return y_str - z * se, y_str + z * se

def stratified_total_estimate(N, y_str):
    """
    Concept: Stratified Population Total Estimate
    Estimates total value across population using stratified sample mean.

    Parameters:
    - N (int): Total population size.
    - y_str (float): Weighted sample mean.
    """
    return N * y_str

def var_total_estimate(Ni, sigma2i, ni):
    """
    Concept: Variance of Total Estimate (Known Variances)
    Estimates variance of total using population variances.

    Parameters:
    - Ni (list): Stratum sizes.
    - sigma2i (list): Known variances for each stratum.
    - ni (list): Sample sizes for each stratum.
    """
    return sum((Ni[i]**2) * ((1 - (ni[i] / Ni[i])) * sigma2i[i] / ni[i]) for i in range(len(Ni)))

def estimated_var_total(Ni, si2, ni):
    """
    Concept: Estimated Variance of Total (Sample Variances)
    Estimates variance of the total population using sample variances si².

    Parameters:
    - Ni (list): Stratum sizes.
    - si2 (list): Sample variances.
    - ni (list): Sample sizes.
    """
    return sum((Ni[i]**2) * ((1 - (ni[i] / Ni[i])) * si2[i] / ni[i]) for i in range(len(Ni)))

def ci_total_estimate(y_T_str, var_y_T, z):
    """
    Concept: Confidence Interval for Total Estimate
    Computes confidence interval for estimated total value across population.

    Parameters:
    - y_T_str (float): Total estimated value.
    - var_y_T (float): Variance of the total estimate.
    - z (float): Z-score for desired confidence level.
    """
    se = math.sqrt(var_y_T)
    return y_T_str - z * se, y_T_str + z * se

def proportional_allocation(Ni, n_total):
    """
    Concept: Proportional Allocation
    Allocates sample size proportionally to each stratum size.

    Parameters:
    - Ni (list): Stratum sizes.
    - n_total (int): Total sample size.
    """
    N = sum(Ni)
    return [round((Ni[i] / N) * n_total) for i in range(len(Ni))]

def neyman_allocation(Ni, sigmai, n_total):
    """
    Concept: Neyman Allocation
    Allocates sample size to strata to minimize variance, using known standard deviations.

    Parameters:
    - Ni (list): Stratum sizes.
    - sigmai (list): Standard deviations for each stratum.
    - n_total (int): Total sample size.
    """
    weights = [Ni[i] * sigmai[i] for i in range(len(Ni))]
    total_weight = sum(weights)
    return [round((weights[i] / total_weight) * n_total) for i in range(len(Ni))]

def optimal_allocation(Ni, sigmai, ci, n_total):
    """
    Concept: Optimal Allocation (with Cost Consideration)
    Allocates sample size using standard deviation and cost per stratum.

    Parameters:
    - Ni (list): Stratum sizes.
    - sigmai (list): Standard deviations.
    - ci (list): Costs per unit in each stratum.
    - n_total (int): Total sample size.
    """
    weights = [Ni[i] * sigmai[i] / ci[i] for i in range(len(Ni))]
    total_weight = sum(weights)
    return [round((weights[i] / total_weight) * n_total) for i in range(len(Ni))]
