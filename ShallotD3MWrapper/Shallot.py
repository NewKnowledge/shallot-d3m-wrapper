import sys
import os.path
import numpy as np
import pandas
import typing
from json import JSONDecoder
from typing import List

from Sloth import Shapelets

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

from .timeseriesloader import TimeSeriesLoaderPrimitive

__author__ = 'Distil'
__version__ = '1.0.1'
__contact__ = 'mailto:jeffrey.gleason@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    shapelet_length = hyperparams.LogUniform(lower = 0, upper = 1, default = 0.1, 
        upper_inclusive = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
       description = 'base shapelet length, expressed as fraction of length of time series')
    num_shapelet_lengths = hyperparams.UniformInt(lower = 1, upper = 100, default = 2, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of different shapelet lengths')
    # default epoch size from https://tslearn.readthedocs.io/en/latest/auto_examples/plot_shapelets.html#sphx-glr-auto-examples-plot-shapelets-py
    epochs = hyperparams.UniformInt(lower = 1, upper = sys.maxsize, default = 200, semantic_types=[
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
       description = 'number of training epochs')
    pass


class Shallot(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
    Produce primitive's classifications for new time series data The input is a numpy ndarray of 
    size (number_of_time_series, time_series_length, dimension) containing new time series. 
    The output is a numpy ndarray containing a predicted class for each of the input time series.
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d351fcf8-5d6c-48d4-8bf6-a56fe11e62d6",
        'version': __version__,
        'name': "shallot",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series', 'Shapelets'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/shallot-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
             'type': metadata_base.PrimitiveInstallationType.PIP,
            'package': 'cython',
            'version': '0.28.5',
        },{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/shallot-d3m-wrapper.git@{git_commit}#egg=ShallotD3MWrapper'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_classification.shapelet_learning.Shallot',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.STOCHASTIC_GRADIENT_DESCENT,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        
        self._params = {}
        self._X_train = None          # training inputs
        self._y_train = None          # training outputs
        self._shapelets = None        # shapelet classifier

    def fit(self) -> None:
        '''
        fits Shapelet classifier using training data from set_training_data and hyperparameters
        '''
        self._shapelets = Shapelets(self._X_train, self._y_train, self.hyperparams['epochs'], 
            self.hyperparams['shapelet_length'], self.hyperparams['num_shapelet_lengths'])

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing training time series

        outputs: numpy ndarray of size (number_time_series,) containing classes of training time series
        '''
        self._X_train = inputs
        self._y_train = outputs


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's classifications for new time series data

        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing new time series 

        Returns
        ----------
        Outputs
            The output is a numpy ndarray containing a predicted class for each of the input time series
        """
        classes = self._shapelets.PredictClasses(inputs)
        return CallResult(classes)


if __name__ == '__main__':
        
    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/66_chlorineConcentration/TRAIN/dataset_TRAIN/datasetDoc.json')
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = {"dataframe_resource":"0"})
    df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)    
    shallot_client = Shallot(hyperparams={'shapelet_length': 0.1,'num_shapelet_lengths': 2, 'epochs':200})
    shallot_client.set_training_data(inputs = df, outputs = 
