# Week 5: Ratio and Regression Estimators

import math

# 1. Population ratio R = mu_y / mu_x
def population_ratio(mu_y, mu_x):
    return mu_y / mu_x

# 2. Sample ratio r = ȳ / x̄
def sample_ratio(y_bar, x_bar):
    return y_bar / x_bar

# 3. Sample ratio from totals r = y_T / x_T
def sample_ratio_from_totals(y_total, x_total):
    return y_total / x_total

# 4. Estimated variance of sample ratio
def estimated_var_ratio(x_bar, r, x, y, n, N):
    f = n / N
    residuals = [(y[i] - r * x[i]) ** 2 for i in range(n)]
    return ((1 - f) / n) * (1 / x_bar**2) * (sum(residuals) / (n - 1))

# 5. CI for sample ratio
def ci_sample_ratio(r, var_r, z):
    se = math.sqrt(var_r)
    return r - z * se, r + z * se

# 6. Ratio estimator of population mean (mu_x known)
def ratio_estimator_mean(r, mu_x):
    return r * mu_x

# 7. Ratio estimator of population total (tau_x known)
def ratio_estimator_total(r, tau_x):
    return r * tau_x

# 8. Variance of ratio estimator for population mean
def var_ratio_estimator_mean(mu_x, var_r):
    return (mu_x**2) * var_r

# --------------------- Regression Estimators -------------------------

# 9. Regression slope beta_hat
def regression_slope(x, y):
    x_bar = sum(x) / len(x)
    y_bar = sum(y) / len(y)
    numerator = sum((x[i] - x_bar) * (y[i] - y_bar) for i in range(len(x)))
    denominator = sum((x[i] - x_bar) ** 2 for i in range(len(x)))
    return numerator / denominator

# 10. Regression intercept alpha_hat
def regression_intercept(x, y):
    x_bar = sum(x) / len(x)
    y_bar = sum(y) / len(y)
    beta = regression_slope(x, y)
    return y_bar - beta * x_bar

# 11. Regression estimator for population mean
def regression_estimator_mean(x, y, mu_x):
    y_bar = sum(y) / len(y)
    x_bar = sum(x) / len(x)
    beta = regression_slope(x, y)
    return y_bar + beta * (mu_x - x_bar)

# 12. Estimated variance of regression estimator
def estimated_var_regression(y, x, alpha_hat, beta_hat, n, N):
    f = n / N
    residuals = [(y[i] - alpha_hat - beta_hat * x[i]) ** 2 for i in range(n)]
    return ((1 - f) / n) * (sum(residuals) / (n - 1))

# Done — All Week 5 formulas are now defined as callable Python functions

