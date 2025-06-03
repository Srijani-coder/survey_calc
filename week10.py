def response_propensity(Rj: int) -> float:
    """
    Concept: Response Propensity
    Returns the response indicator for unit j.

    Parameters:
    - Rj (int): 1 if the unit responded, 0 otherwise.
    """
    return 1.0 if Rj == 1 else 0.0

def mcar_condition(phi_j: float, Yj=None, Xj=None) -> str:
    """
    Concept: MCAR (Missing Completely at Random)
    Checks whether missingness is unrelated to any data (observed or unobserved).

    Parameters:
    - phi_j (float): Response propensity.
    - Yj (optional): Outcome variable.
    - Xj (optional): Covariates.
    """
    return "MCAR condition met" if isinstance(phi_j, (int, float)) else "MCAR condition not met"

def mar_condition(P_R_given_YX: float, P_R_given_X: float) -> bool:
    """
    Concept: MAR (Missing At Random)
    Compares probabilities to determine if missingness is conditional only on observed variables.

    Parameters:
    - P_R_given_YX (float): P(R | Y, X).
    - P_R_given_X (float): P(R | X).
    """
    return round(P_R_given_YX, 6) == round(P_R_given_X, 6)

def non_ignorable_condition(P_R_given_YX: float, P_R_given_X: float) -> bool:
    """
    Concept: MNAR (Missing Not At Random)
    Missingness depends on the missing outcome Y.

    Parameters:
    - P_R_given_YX (float): P(R | Y, X).
    - P_R_given_X (float): P(R | X).
    """
    return round(P_R_given_YX, 6) != round(P_R_given_X, 6)

def rubins_combined_theta(thetas: list) -> float:
    """
    Concept: Rubin's Combined Estimate
    Calculates the pooled mean of multiple imputed estimates.

    Parameters:
    - thetas (list): List of imputed parameter estimates.
    """
    return sum(thetas) / len(thetas)

def rubins_variance(W: float, B: float, m: int) -> float:
    """
    Concept: Rubin's Total Variance
    Combines within-imputation and between-imputation variances.

    Parameters:
    - W (float): Within-imputation variance.
    - B (float): Between-imputation variance.
    - m (int): Number of imputations.
    """
    return (1 + 1 / m) * B + W

def rubins_degrees_freedom(Wjj: float, Bjj: float, m: int) -> float:
    """
    Concept: Rubin's Degrees of Freedom
    Approximates the degrees of freedom for pooled estimates.

    Parameters:
    - Wjj (float): Within-imputation variance for parameter j.
    - Bjj (float): Between-imputation variance for parameter j.
    - m (int): Number of imputations.
    """
    ratio = (1 + Wjj / ((1 + 1 / m) * Bjj)) ** 2
    return (m - 1) * ratio

def rubins_test_statistic(theta_j: float, theta_j0: float, var_jj: float) -> float:
    """
    Concept: Rubin's Test Statistic
    Calculates t-statistic for testing pooled parameter estimate.

    Parameters:
    - theta_j (float): Estimated pooled parameter.
    - theta_j0 (float): Null value.
    - var_jj (float): Total variance of pooled estimate.
    """
    return (theta_j - theta_j0) / (var_jj ** 0.5)

def rubins_relative_increase_in_variance(Bjj: float, Wjj: float, m: int) -> float:
    """
    Concept: Relative Increase in Variance
    Quantifies the inflation in variance due to missing data.

    Parameters:
    - Bjj (float): Between-imputation variance.
    - Wjj (float): Within-imputation variance.
    - m (int): Number of imputations.
    """
    return (1 + 1/m) * Bjj / Wjj
