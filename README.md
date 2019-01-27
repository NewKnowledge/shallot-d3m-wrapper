# Shallot D3M Wrapper
Wrapper of the Shallot Shapelet learning primitive into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater. 

The base Sloth library (which contains the Shapelet class and other time series methods) can be found here: https://github.com/NewKnowledge/sloth

## Install

pip3 install -e git+https://github.com/NewKnowledge/shallot-d3m-wrapper.git#egg=ShallotD3MWrapper --process-dependency-links

## Output
The output is a numpy ndarray containing a predicted class for each of the input time series.

## Available Functions

#### set_training_data

Sets primitive's training data. The inputs are a numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing training time series
and a numpy ndarray of size (number_time_series,) containing classes of training time series. There are no outputs.

#### fit

Fits Shapelet classifier using training data from set_training_data and hyperparameters. There are no inputs or outputs.

#### produce

Produce primitive's classifications for new time series data The input is a numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing new time series. The output is a numpy ndarray containing a predicted class for each of the input time series.
