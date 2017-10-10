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


def scale_standard(raw_data):
    """Wrapper for Scikit-learn StandardScaler : scales raw data and returns stats_df for later rescaling.
    
    :param raw_data: pandas.DataFrame - original DataFrame to scale
    :return: tuple of pandas.DataFrames - (data_scaled, stats_df)
    """
    stats_df = stats(raw_data)
    data_scaled = raw_data.copy()
    data_scaled.iloc[:, :] = StandardScaler().fit_transform(data_scaled)
    return data_scaled, stats_df


def scale_maxabs(raw_data):
    """Wrapper for Scikit-learn MaxAbsScaler : scales raw data and returns stats_df for later rescaling.
    
    :param raw_data: pandas.DataFrame - original DataFrame to scale
    :return: tuple of pandas.DataFrames - (data_scaled, stats_df)
    """
    stats_df = stats(raw_data)
    data_scaled = raw_data.copy()
    data_scaled.iloc[:, :] = MaxAbsScaler().fit_transform(data_scaled)
    return data_scaled, stats_df


def scale_minmax(raw_data):
    """Wrapper for Scikit-learn MinMaxScaler : scales raw data and returns stats_df for later rescaling.
    
    :param raw_data: pandas.DataFrame - original DataFrame to scale
    :return: tuple of pandas.DataFrames - (data_scaled, stats_df)
    """
    stats_df = stats(raw_data)
    data_scaled = raw_data.copy()
    data_scaled.iloc[:, :] = MinMaxScaler().fit_transform(data_scaled)
    return data_scaled, stats_df


def reverse_standard(data_scaled, interest_vars, stats_df):
    """Reverse the Standard scaling of a 2D numpy array (the predicted values) given the place of the predicted features 
    in the original data and the stats DataFrame.
    
    :param data_scaled: numpy.ndarray (2D) - data to transform back to the original scale.
    :param interest_vars: list of ints - indices of the features to predict (indices in the input matrix)
            Example : 321 features as inputs, we predicted the features corresponding to the columns 1, 6 and 315:
            interest_vars is then [1, 6, 315]
    :param stats_df: pandas.DataFrame - DataFrame of coefficients used to reverse scaling (obtained from scale_standard)
    :return: numpy.ndarray (2D) - data transformed back to the original scale.
    """
    data_unscaled = np.copy(data_scaled)
    k = 0
    for i in interest_vars:
        coefs_1 = stats_df["mean"].loc[i]
        coefs_2 = stats_df["std"].loc[i]
        if len(data_unscaled.shape) > 1:
            data_unscaled[:, k] = coefs_1 + coefs_2 * data_unscaled[:, k]
        else:
            data_unscaled = coefs_1 + coefs_2 * data_unscaled
        k = k + 1
    return data_unscaled


def reverse_maxabs(data_scaled, interest_vars, stats_df):
    """Reverse the MaxAbs scaling of a 2D numpy array (the predicted values) given the place of the predicted features 
    in the original data and the stats DataFrame.
    
    :param data_scaled: numpy.ndarray (2D) - data to transform back to the original scale.
    :param interest_vars: list of ints - indices of the features to predict (indices in the input matrix)
            Example : 321 features as inputs, we predicted the features corresponding to the columns 1, 6 and 315:
            interest_vars is then [1, 6, 315]
    :param stats_df: pandas.DataFrame - DataFrame of coefficients used to reverse scaling (obtained from scale_maxabs)
    :return: numpy.ndarray (2D) - data transformed back to the original scale.
    """
    data_unscaled = np.copy(data_scaled)
    k = 0
    for i in interest_vars:
        coefs = stats_df["maxabs"].loc[i]
        if len(data_unscaled.shape) > 1:
            data_unscaled[:, k] = coefs * data_unscaled[:, k]
        else:
            data_unscaled = coefs * data_unscaled
        k = k + 1
    return data_unscaled


def reverse_minmax(data_scaled, interest_vars, stats_df):
    """Reverse the MinMax scaling of a 2D numpy array (the predicted values) given the place of the predicted features 
    in the original data and the stats DataFrame.
    
    :param data_scaled: numpy.ndarray (2D) - data to transform back to the original scale.
    :param interest_vars: list of ints - indices of the features to predict (indices in the input matrix)
            Example : 321 features as inputs, we predicted the features corresponding to the columns 1, 6 and 315:
            interest_vars is then [1, 6, 315]
    :param stats_df: pandas.DataFrame - DataFrame of coefficients used to reverse scaling (obtained from scale_minmax)
    :return: numpy.ndarray (2D) - data transformed back to the original scale.
    """
    data_unscaled = np.copy(data_scaled)
    k = 0
    for i in interest_vars:
        coefs_1 = stats_df["min"].loc[i]
        coefs_2 = stats_df["max"].loc[i]
        if len(data_unscaled.shape) > 1:
            data_unscaled[:, k] = coefs_1 + (coefs_2 - coefs_1) * data_unscaled[:, k]
        else:
            data_unscaled = coefs_1 + (coefs_2 - coefs_1) * data_unscaled
        k = k + 1
    return data_unscaled
