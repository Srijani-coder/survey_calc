def population_parameter():
    """
    Concept: Population Parameter
    - Represents the true value for the entire population (e.g., true % of Republicans in US).
    """
    return float(input("Enter the true population value (e.g., % of Republicans in US): "))

def sample_statistic():
    """
    Concept: Sample Statistic
    - An estimate derived from a sample that approximates the population parameter.
    """
    return float(input("Enter the sample estimate (e.g., % of Republicans in the sample): "))

def sampling_error(population_value, sample_value):
    """
    Concept: Sampling Error
    - The absolute difference between the population parameter and the sample statistic.
    """
    return abs(population_value - sample_value)

def response_rate(num_respondents, total_sampled):
    """
    Concept: Response Rate
    - The proportion (%) of respondents among those sampled.
    """
    return (num_respondents / total_sampled) * 100

def measurement_error(true_value, measured_value):
    """
    Concept: Measurement Error
    - The difference between the true value and the measured value due to inaccuracies.
    """
    return true_value - measured_value

def get_sample_values():
    """
    Utility Function
    - Prompts user to enter sample values separated by commas. Converts input to list of floats.
    - Example: 1, 0, 1, 1, 0
    """
    raw = input("Enter sample values separated by commas (e.g., 1,0,1,0): ")
    return [float(val.strip()) for val in raw.split(',')]

def compute_sample_statistic(values):
    """
    Concept: Compute Sample Statistic
    - Computes the sample mean from a list of numeric values.
    - Useful for estimating proportions or averages in a sample.
    """
    if not values:
        raise ValueError("List of values is empty.")
    return sum(values) / len(values)
