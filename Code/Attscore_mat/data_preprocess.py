import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def calculate_theta_B_for_different_theta_L_values(dataframe, noise_parameter, theta_L_values):
    theta_B_values = []

    for theta_L in theta_L_values:
        # Assuming that the dataframe is a 2D DataFrame
        # You may need to adapt this depending on your actual DataFrame structure
        theta_B = dataframe.applymap(lambda x: x + 1.1 * x * noise_parameter if noise_parameter > 0 else theta_L).mean().mean()
        theta_B_values.append(theta_B)

    return theta_B_values


def add_noise_to_dataframe(dataframe, noise_parameter, theta_L):
    dataframe_with_noise = dataframe.copy()

    for column in dataframe.columns:
        for index in dataframe.index:
            value = dataframe.at[index, column]
            if noise_parameter == 0:
                dataframe_with_noise.at[index, column] = theta_L
            else:
                # Add noise from truncated normal distribution
                theta = value
                sigma = 1.1 * theta * noise_parameter
                noise = np.random.normal(0, sigma)
                dataframe_with_noise.at[index, column] += noise

    return dataframe_with_noise

def fit_curve(theta_L_values, theta_B_values):
    # Fit a piecewise function to the curve
    def piecewise_function(theta_L, m, b, d, c, a, theta_star):
        return np.piecewise(theta_L, [theta_L < theta_star, theta_L >= theta_star],
                            [lambda x: m*x + b, lambda x: ((x - d) / c)**a])

    # Use curve_fit to estimate parameters
    params, _ = curve_fit(piecewise_function, theta_L_values, theta_B_values)
    return params