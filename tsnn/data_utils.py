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
        coefs = stats_df["maxabs"].iloc[i]
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


def inputs_targets_split(data, input_cols, target_cols, samples_length=168, pred_delay=24, pred_length=1):
    """Selects input and target features from data.

    :param data: pandas.DataFrame - loaded data, possibly scaled already.
    :param input_cols: list - names of the columns used as input variables. [] means "all columns".
    :param target_cols: list - names of the columns used as target variables. [] means "all columns".
    :param samples_length: int - time window size (timesteps) for the RNN. Default = 168
    :param pred_delay: int - prediction horizon. We predict values at t+pred_delay. Default = 24
    :param pred_length: int - number of predicted timesteps, starting at t+pred_delay. Default = 1
    :return: Tuple of two pandas.DataFrames -
        - the first is the DataFrame of input features
        - the second is the DataFrame of output features
    """
    inputs = data if len(input_cols) == 0 else data[input_cols]
    targets = data if len(target_cols) == 0 else data[target_cols]
    inp_end = len(data) - (pred_delay + pred_length) + 1
    tar_start = samples_length + pred_delay + pred_length - 2
    inputs = inputs.iloc[:inp_end]
    targets = targets.iloc[tar_start:].reset_index(drop=True)
    return inputs, targets


def train_val_split(targets, train_ratio=0.6, val_ratio=0.2):
    """Returns range limits for Train / Validation / Test sets
    
    :param targets: pandas.DataFrame - target columns (obtained after scaling and inputs_targets_split) 
    :param train_ratio: float in ]0, 1[ - Proportion of rows to use as training set. Default = 0.6
    :param val_ratio: float in ]0, 1[ - Proportion of rows to use as validation set. Default = 0.2
    :param pred_delay: int - prediction horizon > 0. We predict values at t+pred_delay. Default = 24
    :param pred_length: int - number of predicted timesteps, starting at t+pred_delay. Default = 1
    :return: tuple of 3 int-tuples - (train_start, train_end), (val_start, val_end), (test_start, test_end)
    """
    train_start = 0
    train_end = round(train_ratio * len(targets))
    val_start = train_end
    val_end = val_start + round(val_ratio * len(targets))
    test_start = val_end
    test_end = len(targets)
    return (train_start, train_end), (val_start, val_end), (test_start, test_end)


def sample_gen_rnn(scaled_inputs,
                   scaled_targets,
                   limits=(None, None),
                   samples_length=168,
                   sampling_step=1,
                   batch_size=24,
                   inputs_only=False):
    """Batch generator for RNN architectures.

    :param scaled_inputs: pandas.DataFrame - inputs obtained from inputs_targets_split()
    :param scaled_targets: pandas.DataFrame - targets obtained from inputs_targets_split()
    :param limits: int tuple - (start_row, end_row) : start and end rows, one of train_val_split() results
    :param samples_length: int - time window size (timesteps) for the RNN. Default = 168
    :param sampling_step: int - stride used when extracting time windows. Default = 1 (no position skipped)
    :param batch_size: int - batch size for mini-batch gradient descent. Default = 24
    :param inputs_only: boolean - whether the generator yields inputs only or (inputs, targets). 
        - Set to False if using fit_generator or evaluate_generator
        - Set to True if using predict_generator
    :yield: tuple - (input_batch, target_batch)
    """
    if limits[0] is None:
        limits[0] = 0
    if limits[1] is None:
        limits[1] = len(scaled_targets)

    inp_row = limits[0]
    tar_row = limits[0]
    inp_batch = []
    tar_batch = []

    while inp_row < limits[1]:
        inp = scaled_inputs.iloc[inp_row:inp_row + samples_length].values
        inp_batch.append(inp)
        tar = scaled_targets.iloc[tar_row].values
        tar_batch.append(tar)

        if len(inp_batch) == batch_size or inp_row == (limits[1] - 1):
            if inputs_only:
                yield np.array(inp_batch)
            else:
                yield np.array(inp_batch), np.array(tar_batch)
            inp_batch = []
            tar_batch = []
        inp_row += sampling_step
        tar_row += sampling_step

        if inp_row >= limits[1]:
            inp_row = limits[0]
            tar_row = limits[0]


###### GENERATEUR pour les X seuls