import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler


def stats(raw_data):
    """Computes statistics useful for reverse scaling of predictions.
    
    :param raw_data: pandas.DataFrame - original DataFrame
    :return: pandas.DataFrame - DataFrame of coefficients used for reverse scaling
    """
    stats_df = raw_data.describe().transpose()
    stats_df = stats_df[["min", "max", "mean", "std"]]
    stats_df["maxabs"] = pd.concat([stats_df["min"].abs(), stats_df["max"].abs()], axis=1).max(axis=1)
    stats_df["std"] = np.std(raw_data)
    return stats_df


def prepare_scaling(raw_data, method):
    """Creates scaler object and prepares coefficient to reverse scaling later
    
    :param raw_data: pandas.DataFrame - original DataFrame to scale
    :param method: String in ["", "standard", "minmax", "maxabs"] - method to use for scaling. If "", no scaling.
    :return: tuple of two elements -
        - Scaler object : one of [None, StandardScaler(), MaxAbsScaler(), MinMaxScaler()]
        - DataFrame of coefficients used for reverse scaling
    """
    stats_df = stats(raw_data)
    scalers = {
        "": None,
        "standard": StandardScaler(),
        "maxabs": MaxAbsScaler(),
        "minmax": MinMaxScaler()
    }
    scaler = scalers[method]
    return scaler, stats_df


def scale(raw_data, scaler):
    """Wrapper for Scikit-learn scalers; : scales raw data with the chosen scaler.
    
    :param raw_data: pandas.DataFrame - original DataFrame to scale
    :param scaler: scaler object from prepare_scaling
    :return: pandas.DataFrames - the scaled data
    """
    data_scaled = raw_data.copy()
    if scaler is not None:
        data_scaled.iloc[:, :] = scaler.fit_transform(data_scaled)
    return data_scaled


def reverse_scaling(data_scaled, interest_cols, stats_df, method):
    """Reverse the scaling of the columns of a 2D numpy array (the predicted values) given the place of these columns 
    in the original data, the coefficients DataFrame and the scaling method used.

    :param data_scaled: numpy.ndarray (2D) - data to transform back to the original scale.
    :param interest_cols: list of ints - places of data_scaled columns in the original data
    :param stats_df: pandas.DataFrame - DataFrame of coefficients used to reverse scaling (obtained from scale())
    :param method: String in ["", "standard", "minmax", "maxabs"] - method to use for scaling. If "", no scaling.
    :return: numpy.ndarray (2D) - data transformed to the original scale.
    """
    data_unscaled = np.copy(data_scaled)
    if method == "":
        pass
    else:
        k = 0
        for i in interest_cols:
            if method == "maxabs":
                coefs = stats_df["maxabs"].loc[i]
                if len(data_unscaled.shape) > 1:
                    data_unscaled[:, k] = coefs * data_unscaled[:, k]
                else:
                    data_unscaled = coefs * data_unscaled
            if method == "standard":
                coefs_1 = stats_df["mean"].loc[i]
                coefs_2 = stats_df["std"].loc[i]
                if len(data_unscaled.shape) > 1:
                    data_unscaled[:, k] = coefs_1 + coefs_2 * data_unscaled[:, k]
                else:
                    data_unscaled = coefs_1 + coefs_2 * data_unscaled
            if method == "minmax":
                coefs_1 = stats_df["min"].loc[i]
                coefs_2 = stats_df["max"].loc[i]
                if len(data_unscaled.shape) > 1:
                    data_unscaled[:, k] = coefs_1 + (coefs_2 - coefs_1) * data_unscaled[:, k]
                else:
                    data_unscaled = coefs_1 + (coefs_2 - coefs_1) * data_unscaled
            k = k + 1
    return data_unscaled
