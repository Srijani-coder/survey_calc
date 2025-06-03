import math

# Week 6: Cluster Sampling (Equal and Unequal Sizes)

def cluster_mean(cluster_data):
    """
    Concept: Cluster Mean
    Computes the mean within a single cluster.

    Parameters:
    - cluster_data (list): Values within a single cluster.
    """
    return sum(cluster_data) / len(cluster_data)

def cluster_variance(cluster_data):
    """
    Concept: Cluster Variance
    Measures variability within a single cluster.

    Parameters:
    - cluster_data (list): Values within a single cluster.
    """
    L = len(cluster_data)
    mean = cluster_mean(cluster_data)
    return sum((x - mean) ** 2 for x in cluster_data) / (L - 1)

def overall_cluster_sample_mean(cluster_means):
    """
    Concept: Overall Cluster Sample Mean
    Calculates the mean across all sampled clusters.

    Parameters:
    - cluster_means (list): Mean of each sampled cluster.
    """
    return sum(cluster_means) / len(cluster_means)

def estimated_var_ybar_CL_equal(cluster_means, N, n):
    """
    Concept: Variance of Cluster Mean (Equal-sized Clusters)
    Estimates the variance of the cluster sample mean assuming equal-sized clusters.

    Parameters:
    - cluster_means (list): List of means of each cluster.
    - N (int): Total number of clusters in population.
    - n (int): Number of sampled clusters.
    """
    ybar_CL = overall_cluster_sample_mean(cluster_means)
    f = n / N
    squared_diffs = [(y_i - ybar_CL) ** 2 for y_i in cluster_means]
    return (1 - f) * sum(squared_diffs) / ((n - 1) * n)

def ci_cluster_mean_equal(ybar_CL, var_estimate, z):
    """
    Concept: Confidence Interval for Cluster Mean (Equal-sized Clusters)
    Constructs CI for mean when clusters are of equal size.

    Parameters:
    - ybar_CL (float): Sample cluster mean.
    - var_estimate (float): Variance estimate of the mean.
    - z (float): Z-score for confidence level.
    """
    se = math.sqrt(var_estimate)
    return ybar_CL - z * se, ybar_CL + z * se

def weighted_cluster_sample_mean(mi, ybar_i):
    """
    Concept: Weighted Cluster Mean (Unequal-sized Clusters)
    Computes overall mean using cluster means weighted by cluster sizes.

    Parameters:
    - mi (list): Cluster sizes.
    - ybar_i (list): Cluster means.
    """
    total_m = sum(mi)
    return sum(mi[i] * ybar_i[i] for i in range(len(mi))) / total_m

def estimated_var_ybar_CL_unequal(mi, ybar_i, ybar_CL, N, n):
    """
    Concept: Variance of Cluster Mean (Unequal-sized Clusters)
    Estimates the variance for cluster sample mean with unequal-sized clusters.

    Parameters:
    - mi (list): Cluster sizes.
    - ybar_i (list): Cluster means.
    - ybar_CL (float): Overall weighted cluster mean.
    - N (int): Number of clusters in population.
    - n (int): Number of clusters sampled.
    """
    f = n / N
    total_m = sum(mi)
    squared_terms = [(mi[i] * (ybar_i[i] - ybar_CL)) ** 2 for i in range(n)]
    return (1 - f) / n * (N**2 / total_m**2) * sum(squared_terms) / (n - 1)

def ci_cluster_mean_unequal(ybar_CL, var_estimate, z):
    """
    Concept: Confidence Interval for Cluster Mean (Unequal-sized Clusters)
    Constructs CI for the mean when clusters have different sizes.

    Parameters:
    - ybar_CL (float): Weighted cluster mean.
    - var_estimate (float): Estimated variance.
    - z (float): Z-score for CI.
    """
    se = math.sqrt(var_estimate)
    return ybar_CL - z * se, ybar_CL + z * se

def estimated_average_cluster_size(mi):
    """
    Concept: Average Cluster Size
    Calculates the average size of sampled clusters.

    Parameters:
    - mi (list): Sizes of each sampled cluster.
    """
    return sum(mi) / len(mi)
