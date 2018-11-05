# Shallot D3M Wrapper
Shallot - SHApeLet Learning Over Time-series

This library is a wrapper of Sloth's Shapelets class into the D3M infrastructure. Code is written in Python 3.6.

## Available Functions

# set_training_data

Sets primitive's training data. The inputs are a numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing training time series
and a numpy ndarray of size (number_time_series,) containing classes of training time series. There are no outputs.

# fit

Fits Shapelet classifier using training data from set_training_data and hyperparameters. There are no inputs or outputs.

# produce

Produce primitive's classifications for new time series data The input is a numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing new time series. The output is a numpy ndarray containing a predicted class for each of the input time series.